# transforms with load and store callbacks, see `callbacks.jl`

const cufftCallbackNumber = Union{Float32,Float64,ComplexF32,ComplexF64}

"""
    callback_issue(p::CuFFTPlan) -> Union{Nothing,String}

Why the transform `p` cannot be executed with callbacks on the current device and stream,
or `nothing` if it can.
"""
function callback_issue(p)
    T, S = output_type(p), input_type(p)
    T <: cufftCallbackNumber && S <: cufftCallbackNumber ||
        return "callbacks only support single and double precision"
    is_inplace(p) && return "in-place plans load from the array they store to"
    # callbacks index the arrays as the user sees them
    p.perm === nothing || return "cuFFT cannot batch this transform without permuting it"
    version() >= v"12" || return "callbacks need cuFFT 12 (CUDA 13) or newer"
    CUDACore.compiler_version() >= v"13.5" ||
        return "callbacks are compiled with libNVVM from CUDA 13.5 or newer"
    # cuFFT links callbacks with the nvJitLink from the compiler, which has to be at least as
    # new as cuFFT itself
    CUDACore.compiler_version() >= CUDACore.runtime_version() ||
        return "the CUDA compiler's nvJitLink is older than cuFFT"
    # for older GPUs, libNVVM only reads the IR Julia generates with `-nvvm-latest`, and
    # nvJitLink then needs that option too, which cuFFT doesn't pass
    capability(device()) >= v"10.0" ||
        return "callbacks need a GPU with compute capability 10.0 (Blackwell) or newer"
    real_size = S <: Real ? p.input_size : p.output_size
    all(d -> largest_prime_factor(real_size[d]) < 127, p.region) ||
        return "cuFFT doesn't support callbacks for sizes with a prime factor of 127 or larger"
    # the callbacks' state lives in memory owned by the handle, which a graph wouldn't keep
    # alive, and the ordering of executions wouldn't cover the graph's launches
    is_capturing(stream()) && return "transforms with callbacks cannot be captured into a graph"
    return nothing
end

is_inplace(::CuFFTPlan{<:Any,<:Any,<:Any,inplace}) where {inplace} = inplace

function largest_prime_factor(n::Int)
    largest, d = 1, 2
    while d * d <= n
        while n % d == 0
            largest = d
            n ÷= d
        end
        d += 1
    end
    return max(largest, n)
end

"""
    execute_fused!(p::CuFFTPlan, input, output; load=nothing, store=nothing)

Enqueue the out-of-place transform `p` with callbacks: `load(x, i)` provides input element
`i` instead of `input`, and `store(y, i, v)` consumes output element `v` at index `i`.
`input` can be `nothing` when there is a load callback, which cuFFT then calls instead of
reading its input. `output` always has to be an array of the plan's output type and size,
even with a store callback that writes elsewhere: cuFFT may use it as scratch space, so
its contents are undefined afterwards unless the store callback writes them.

Returns `nothing`, or the reason why nothing was executed: a callback that does not compile
(e.g., as it may throw) or is not a bits value after `cudaconvert`, or a transform for
which cuFFT doesn't call callbacks. The caller has to check [`callback_issue`](@ref) first.
"""
function execute_fused!(p::CuFFTPlan{T,S,K,false,N}, input::Union{Nothing,DenseCuArray{S}},
                        output::DenseCuArray{T}; load=nothing, store=nothing) where {T,S,K,N}
    input === nothing && load === nothing &&
        throw(ArgumentError("a transform without input needs a load callback"))
    L = compiled_wrapper(LoadWrapper, load, S, N)
    St = compiled_wrapper(StoreWrapper, store, T, N)
    (L === missing || St === missing) &&
        return "a callback cannot be compiled, as it may throw or its state is not a bits value"
    callbacks_honored(p, load !== nothing, store !== nothing) ||
        return "cuFFT doesn't call callbacks for this transform"
    submit_fused!(p, input, output, L, St, load, store)
    return
end

function submit_fused!(p::CuFFTPlan{T,S,K}, input, output, L, St, load, store) where {T,S,K}
    real_size = S <: Real ? p.input_size : p.output_size
    h = get_callback_plan(T, S, real_size, p.region, L, St)
    try
        s = stream()
        cufftSetStream(h, s)
        # the handle is only ours until it is released, while the transform keeps running
        CUDACore.wait(h.done, s)

        # convert the callbacks like kernel arguments, so that the memory they capture is
        # used on this stream
        managed = CUDACore.Managed[]
        load === nothing ||
            upload_state!(h.load_info, CallbackState(cudaconvert(load, managed), p.input_size), s)
        store === nothing ||
            upload_state!(h.store_info, CallbackState(cudaconvert(store, managed), p.output_size), s)
        CUDACore.with_managed(managed; stream=s) do
            # cuFFT doesn't dereference the input when it has a load callback
            src = input === nothing ? reinterpret(CuPtr{Cvoid}, pointer(h.load_info)) : input
            if h.workspace_size == 0
                cufftXtExec(h, src, output, K)
            else
                with_workspace(h.workspace_size) do workspace
                    cufftSetWorkArea(h, workspace)
                    cufftXtExec(h, src, output, K)
                end
            end
        end
        record(h.done, s)
    finally
        release_callback_plan(h)
    end
    return
end

# cuFFT accepts callbacks for every plan, but some of its kernels don't call them, e.g.
# those for large batches of small 2D complex transforms (as of cuFFT 12.5). It doesn't
# report which, so every kind of transform is tried once with callbacks whose effect is
# known: a load callback that returns zeros for an input of ones, and a store callback that
# marks the elements it is called for.
struct ProbeLoad{S} end
@inline (::ProbeLoad{S})(x, i) where {S} = zero(S)
struct ProbeStore{A}
    seen::A
end
CUDACore.Adapt.adapt_structure(to, s::ProbeStore) = ProbeStore(CUDACore.Adapt.adapt(to, s.seen))
@inline function (s::ProbeStore)(y, i, v)
    @inbounds s.seen[i] = iszero(v) ? 0x01 : 0x02
    return
end

const honored = Dict{Any,Bool}()
const honored_lock = ReentrantLock()

function callbacks_honored(p::CuFFTPlan{T,S}, has_load::Bool, has_store::Bool) where {T,S}
    key = (context(), T, S, p.input_size, p.region, has_load, has_store)
    @lock honored_lock get!(() -> probe_callbacks(p, has_load, has_store), honored, key)
end

function probe_callbacks(p::CuFFTPlan{T,S,K,false,N}, has_load, has_store) where {T,S,K,N}
    x = CUDACore.fill(has_load ? one(S) : zero(S), p.input_size)
    y = CuArray{T}(undef, p.output_size)
    seen = CUDACore.zeros(UInt8, p.output_size)
    load = has_load ? ProbeLoad{S}() : nothing
    store = has_store ? ProbeStore(seen) : nothing
    L = compiled_wrapper(LoadWrapper, load, S, N)
    St = compiled_wrapper(StoreWrapper, store, T, N)
    (L === missing || St === missing) && return false
    submit_fused!(p, x, y, L, St, load, store)
    called = has_store ? all(==(0x01), seen) : all(iszero, y)
    foreach(unsafe_free!, (x, y, seen))
    return called
end

function upload_state!(info::CUDACore.DeviceMemory, state::S, stream) where {S}
    ref = Ref(state)
    GC.@preserve ref begin
        # copies from pageable memory are staged before the call returns
        unsafe_copyto!(reinterpret(CuPtr{S}, pointer(info)),
                       Base.unsafe_convert(Ptr{S}, ref), 1; async=true, stream)
    end
    return
end

# whether broadcasts around lazy transforms are fused into them, for testing the fallback
const fuse_broadcasts = Ref(true)

# (logged here so that `JULIA_DEBUG=cuFFT` shows it)
unfused(reason) = @debug "cuFFT: not fusing a broadcast into its transform" reason

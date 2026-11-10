# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft, brfft, irfft,
    Plan, ScaledPlan

using LinearAlgebra

input_type(plan::ScaledPlan) = input_type(plan.p)
output_type(plan::ScaledPlan) = output_type(plan.p)

# broadcast instead of `rmul!`, which falls back to scalar indexing when the in-place plan
# was applied to a view
function Base.:(*)(p::ScaledPlan, x::StridedCuArray)
    y = p.p * x
    y .*= p.scale
end

## plan structure

# T is the output type
# S is the input ("source") type

# K is an integer flag for forward/backward
# also used as an alias for r2c/c2r

# inplace is a boolean flag

# N is the number of dimensions

mutable struct CuFFTPlan{T<:cufftNumber,S<:cufftNumber,K,inplace,N,R} <: Plan{S}
    # handle to Cuda low level plan. Note that this plan sometimes has lower dimensions
    # to handle more transform cases such as individual directions
    handle::cufftHandle
    ctx::CuContext
    input_size::NTuple{N,Int}   # Julia size of input array
    output_size::NTuple{N,Int}  # Julia size of output array
    region::NTuple{R,Int}
    # permutation of the dimensions into a layout cuFFT can batch, or `nothing`
    perm::Union{Nothing,NTuple{N,Int}}
    workspace_size::Int
    # a plan can be shared by tasks, which execute it on their own streams. this lock
    # serializes configuring the handle's stream and work area with enqueueing the
    # transform; every execution uses its own work area, so the kernels may overlap.
    lock::ReentrantLock
    pinv::ScaledPlan{T}         # required by AbstractFFTs API, will be defined by AbstractFFTs if needed

    function CuFFTPlan{T,S,K,inplace,N,R}(input_size::NTuple{N,Int}, output_size::NTuple{N,Int},
                                          region::NTuple{R,Int}
                                          ) where {T<:cufftNumber,S<:cufftNumber,K,inplace,N,R}
        abs(K) == 1 || throw(ArgumentError("FFT direction must be either -1 (forward) or +1 (inverse)"))
        inplace isa Bool || throw(ArgumentError("FFT inplace argument must be a Bool"))

        # the real side of a real transform determines the transform lengths
        real_size = S <: Real ? input_size : output_size
        perm = staging_permutation(real_size, region)
        handle = if perm === nothing
            get_plan(T, S, real_size, region)
        else
            get_plan(T, S, permute(real_size, perm), map(d -> findfirst(==(d), perm)::Int, region))
        end

        workspace_size = Ref{Csize_t}()
        cufftGetSize(handle, workspace_size)
        p = new{T,S,K,inplace,N,R}(handle, context(), input_size, output_size, region, perm,
                                   Int(workspace_size[]), ReentrantLock())
        resource_finalizer(unsafe_free!, p; blocking=false)
        p
    end
end

function CUDACore.unsafe_free!(plan::CuFFTPlan)
    if plan.handle != C_NULL
        context!(plan.ctx) do
            release_plan(plan.handle)
        end
        plan.handle = C_NULL
    end
end

function showfftdims(io, sz, T)
    if isempty(sz)
        print(io,"0-dimensional")
    elseif length(sz) == 1
        print(io, sz[1], "-element")
    else
        print(io, join(sz, "×"))
    end
    print(io, " CuArray of ", T)
end

function Base.show(io::IO, p::CuFFTPlan{T,S,K,inplace}) where {T,S,K,inplace}
    print(io, "CUFFT ",
          inplace ? "in-place " : "",
          S == T ? "$T " : "$(S)-to-$(T) ",
          K == CUFFT_FORWARD ? "forward " : "backward ",
          "plan for ")
    showfftdims(io, p.input_size, S)
end

output_type(::CuFFTPlan{T,S}) where {T,S} = T
input_type(::CuFFTPlan{T,S}) where {T,S} = S

# for some reason, cufftHandle is an integer and not a pointer...
Base.convert(::Type{cufftHandle}, p::CuFFTPlan) = p.handle
# we also need to be able to convert CuFFTPlans that have been wrapped in a ScaledPlan
Base.convert(::Type{cufftHandle}, p::ScaledPlan{T,P}) where {T,P<:CuFFTPlan} = convert(cufftHandle, p.p)

Base.size(p::CuFFTPlan) = p.input_size

## plan methods

# AbstractFFTs.jl passes planner keyword arguments, such as FFTW's `flags` and `timelimit`,
# on to `plan_*` methods. cuFFT has no such options, so they are accepted and ignored. That
# includes FFTW.PRESERVE_INPUT, which holds anyway: out-of-place complex-to-real transforms
# operate on a copy of the input. Every method needs the `kwargs...`: a method without
# keyword arguments is not considered for a call that passes some, which then ends up in a
# less specific method that does, e.g., FFTW's or the generic ones from AbstractFFTs.jl.

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::StridedCuArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::StridedCuArray{<:Real}, region; kwargs...) = $pf(complexfloat(x), region; kwargs...)
        $f(x::StridedCuArray{<:Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::StridedCuArray{<:Complex{<:Union{Integer,Rational}}}, region; kwargs...) = $pf(complexfloat(x), region; kwargs...)
    end
end
rfft(x::StridedCuArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(realfloat(x), region)
plan_rfft(x::StridedCuArray{<:Real}, region; kwargs...) = plan_rfft(realfloat(x), region; kwargs...)

for f in (:brfft, :irfft)
    @eval begin
        $f(x::StridedCuArray{<:Real}, d::Integer, region=1:ndims(x)) = $f(complexfloat(x), d, region)
        $f(x::StridedCuArray{<:Complex{<:Union{Integer,Rational}}}, d::Integer, region=1:ndims(x)) = $f(complexfloat(x), d, region)
    end
end

# layouts that cuFFT can't batch (see `batchable`) are staged in a permuted layout, with the
# transformed dimensions first if dimension 1 is transformed, and last if not, so that the
# permutation keeps dimension 1 in place where possible.
function staging_permutation(sz::Dims{N}, region::Dims) where N
    batchable(sz, region) && return nothing
    other_dims = Tuple(d for d in 1:N if d ∉ region)
    perm = 1 in region ? (region..., other_dims...) : (other_dims..., region...)
    return perm::Dims{N}
end

permute(sz::Dims{N}, perm::Dims{N}) where N = ntuple(i -> sz[perm[i]], N)

function ensure_unique(s::NTuple{N, Int}) where N
    for i in 1:N, j in i+1:N
        s[i] == s[j] && throw(ArgumentError(
            "FFT region dimensions must be unique; got $s"))
    end
    s
end

# rfft/brfft cannot reorder region: cuFFT halves region[1], and AbstractFFTs
# uses `first(region)` to size the output (definitions.jl:343,349). So the
# user must already supply region in strictly increasing order.
function ensure_strictly_increasing(s::NTuple{N, Int}) where N
    for i in 1:N-1
        s[i] >= s[i+1] && throw(ArgumentError(
            "for rfft/brfft, region must be in strictly increasing order " *
            "(its first element is the dimension reduced from N to N÷2+1); got $s"))
    end
    s
end

# Sort on tuples is only implemented as of Julia 1.12, and cuFFT supports at most
# three transform dimensions per plan, so we hand-code the cases here.
ensure_increasing(s::NTuple{1, Int}) = s
ensure_increasing(s::NTuple{2, Int}) = s[1] > s[2] ? (s[2], s[1]) : s
function ensure_increasing(s::NTuple{3, Int})
    s[1] > s[2] && (s = (s[2], s[1], s[3]))
    s[2] > s[3] && (s = (s[1], s[3], s[2]))
    s[1] > s[2] && (s = (s[2], s[1], s[3]))
    s
end
function ensure_increasing(s::NTuple{N, Int}) where N
    throw(ArgumentError("cuFFT supports at most 3 transform dimensions per plan; got $N"))
end
# region is an iterable subset of dimensions
# spec. an integer, range, tuple, or array

# try to constant-propagate the `region` argument when it is not a tuple. This helps with
# inference of calls like plan_fft(X), which is translated by AbstractFFTs.jl into
# plan_fft(X, 1:ndims(X)).
for f in (:plan_fft!, :plan_bfft!, :plan_fft, :plan_bfft)
    @eval begin
        Base.@constprop :aggressive function $f(X::StridedCuArray{T,N}, region; kwargs...) where {T<:cufftComplexes,N}
            R = length(region)
            region = NTuple{R,Int}(region)
            $f(X, region; kwargs...)
        end
    end
end

# inplace complex
function plan_fft!(X::StridedCuArray{T,N}, region::NTuple{R,Int}; kwargs...) where {T<:cufftComplexes,N,R}
    K = CUFFT_FORWARD
    inplace = true
    region = ensure_increasing(ensure_unique(region))

    CuFFTPlan{T,T,K,inplace,N,R}(size(X), size(X), region)
end

function plan_bfft!(X::StridedCuArray{T,N}, region::NTuple{R,Int}; kwargs...) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = true
    region = ensure_increasing(ensure_unique(region))

    CuFFTPlan{T,T,K,inplace,N,R}(size(X), size(X), region)
end

# out-of-place complex
function plan_fft(X::StridedCuArray{T,N}, region::NTuple{R,Int}; kwargs...) where {T<:cufftComplexes,N,R}
    K = CUFFT_FORWARD
    inplace = false
    region = ensure_increasing(ensure_unique(region))

    CuFFTPlan{T,T,K,inplace,N,R}(size(X), size(X), region)
end

function plan_bfft(X::StridedCuArray{T,N}, region::NTuple{R,Int}; kwargs...) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = false
    region = ensure_increasing(ensure_unique(region))

    CuFFTPlan{T,T,K,inplace,N,R}(size(X), size(X), region)
end

# out-of-place real-to-complex
Base.@constprop :aggressive function plan_rfft(X::StridedCuArray{T,N}, region; kwargs...) where {T<:cufftReals,N}
    R = length(region)
    region = NTuple{R,Int}(region)
    plan_rfft(X, region; kwargs...)
end

function plan_rfft(X::StridedCuArray{T,N}, region::NTuple{R,Int}; kwargs...) where {T<:cufftReals,N,R}
    K = CUFFT_FORWARD
    inplace = false
    region = ensure_strictly_increasing(region)

    xdims = size(X)
    ydims = Base.setindex(xdims, div(xdims[region[1]], 2) + 1, region[1])

    CuFFTPlan{complex(T),T,K,inplace,N,R}(xdims, ydims, region)
end

# out-of-place complex-to-real
Base.@constprop :aggressive function plan_brfft(X::StridedCuArray{T,N}, d::Integer, region; kwargs...) where {T<:cufftComplexes,N}
    R = length(region)
    region = NTuple{R,Int}(region)
    plan_brfft(X, d, region; kwargs...)
end

function plan_brfft(X::StridedCuArray{T,N}, d::Integer, region::NTuple{R,Int}; kwargs...) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = false
    region = ensure_strictly_increasing(region)

    xdims = size(X)
    ydims = Base.setindex(xdims, d, region[1])

    CuFFTPlan{real(T),T,K,inplace,N,R}(xdims, ydims, region)
end


function plan_inv(p::CuFFTPlan{T,S,CUFFT_INVERSE,inplace,N,R}
                  ) where {T<:cufftNumber,S<:cufftNumber,inplace,N,R}
    ScaledPlan(CuFFTPlan{S,T,CUFFT_FORWARD,inplace,N,R}(p.output_size, p.input_size, p.region),
               normalization(real(T), p.output_size, p.region))
end

function plan_inv(p::CuFFTPlan{T,S,CUFFT_FORWARD,inplace,N,R}
                  ) where {T<:cufftNumber,S<:cufftNumber,inplace,N,R}
    ScaledPlan(CuFFTPlan{S,T,CUFFT_INVERSE,inplace,N,R}(p.output_size, p.input_size, p.region),
               normalization(real(S), p.input_size, p.region))
end


## plan execution

# NOTE: "in-place complex-to-real FFTs may overwrite arbitrary imaginary input point values
#       [...]. Out-of-place complex-to-real FFT will always overwrite input buffer."
#       see # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055.

function assert_applicable(p::CuFFTPlan{T,S}, X::StridedCuArray) where {T,S}
    (size(X) == p.input_size) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,S,K,inplace}, X::StridedCuArray{S},
                           Y::StridedCuArray{T}) where {T,S,K,inplace}
    assert_applicable(p, X)
    if size(Y) != p.output_size
        throw(ArgumentError("CuFFT plan applied to wrong-size output"))
    elseif inplace != (pointer(X) == pointer(Y))
        throw(ArgumentError(string("CuFFT ",
                                   inplace ? "in-place" : "out-of-place",
                                   " plan applied to ",
                                   inplace ? "out-of-place" : "in-place",
                                   " data")))
    elseif inplace && strides(X) != strides(Y)
        throw(ArgumentError("CuFFT in-place plan applied to arrays with different strides"))
    end
end


# Enqueue the transform of `x` into `y` on the current stream. Both need to be laid out as
# the cuFFT plan expects, i.e., permuted if the plan stages its data.
function execute!(plan::CuFFTPlan{T,S,K}, x::DenseCuArray{S}, y::DenseCuArray{T}) where {T,S,K}
    @lock plan.lock begin
        cufftSetStream(plan, stream())
        if plan.workspace_size == 0
            cufftXtExec(plan, x, y, K)
        else
            with_workspace(plan.workspace_size) do workspace
                cufftSetWorkArea(plan, workspace)
                cufftXtExec(plan, x, y, K)
            end
        end
    end
end


## high-level integrations

# cuFFT plans describe dense arrays, so strided views are staged through dense buffers, as
# are layouts that cuFFT cannot batch (see `staging_permutation`).
function transform!(y::StridedCuArray{T}, p::CuFFTPlan{T,S,K,inplace}, x::StridedCuArray
                   ) where {T,S,K,inplace}
    perm = p.perm
    if perm === nothing
        if inplace
            z = x isa DenseCuArray ? x : copy1(S, x)
            w = z
        else
            # out-of-place complex-to-real transforms overwrite their input, and inputs that
            # may overlap the output need a snapshot
            z = T<:Real || !(x isa DenseCuArray{S}) || Base.mightalias(x, y) ? copy1(S, x) : x
            w = y isa DenseCuArray ? y : CuArray{T}(undef, p.output_size)
        end
        execute!(p, z, w)
        w === y || copyto!(y, w)
    else
        # the permuted copy can be overwritten, and doesn't alias the output
        x′ = x isa CuArray{S} ? x : copy1(S, x)
        z = permutedims(x′, perm)
        x′ === x || unsafe_free!(x′)
        w = inplace ? z : CuArray{T}(undef, permute(p.output_size, perm))
        execute!(p, z, w)
        if y isa CuArray
            permutedims!(y, w, invperm(perm))
        else
            y′ = permutedims(w, invperm(perm))
            copyto!(y, y′)
            unsafe_free!(y′)
        end
    end
    z === x || unsafe_free!(z)
    w === y || w === z || unsafe_free!(w)
    y
end

function LinearAlgebra.mul!(y::StridedCuArray{T}, p::CuFFTPlan{T,S,K,inplace}, x::StridedCuArray{S}
                           ) where {T,S,K,inplace}
    assert_applicable(p, x, y)
    transform!(y, p, x)
end

# AbstractFFTs.jl scales with `lmul!`, which falls back to scalar indexing on views
function LinearAlgebra.mul!(y::StridedCuArray, p::ScaledPlan{T,P}, x::StridedCuArray
                           ) where {T,P<:CuFFTPlan}
    mul!(y, p.p, x)
    y .*= p.scale
end

Base.:(*)(p::CuFFTPlan{T,S,K,true}, x::StridedCuArray{S}) where {T,S,K} = mul!(x, p, x)

# out-of-place plans also accept inputs of another element type, which are converted
function Base.:(*)(p::CuFFTPlan{T,S,K,false}, x::StridedCuArray{S1,M}) where {T,S,K,S1,M}
    assert_applicable(p, x)
    transform!(CuArray{T,M}(undef, p.output_size), p, x)
end

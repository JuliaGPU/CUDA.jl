# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft, irfft,
    Plan, ScaledPlan

using LinearAlgebra

input_type(plan::ScaledPlan) = input_type(plan.p)
output_type(plan::ScaledPlan) = output_type(plan.p)

Base.:(*)(p::ScaledPlan, x::DenseCuArray) = rmul!(p.p * x, p.scale)

## plan structure

# T is the output type
# S is the input ("source") type

# K is an integer flag for forward/backward
# also used as an alias for r2c/c2r

# inplace is a boolean flag

# N is the number of dimensions

mutable struct CuFFTPlan{T<:cufftNumber,S<:cufftNumber,K,inplace,N,R,B} <: Plan{S}
    # handle to Cuda low level plan. Note that this plan sometimes has lower dimensions
    # to handle more transform cases such as individual directions
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    input_size::NTuple{N,Int}   # Julia size of input array
    output_size::NTuple{N,Int}  # Julia size of output array
    region::NTuple{R,Int}
    buffer::B                   # buffer for out-of-place complex-to-real FFT, or `nothing` if not needed
    pinv::ScaledPlan{T}         # required by AbstractFFTs API, will be defined by AbstractFFTs if needed

    function CuFFTPlan{T,S,K,inplace,N,R,B}(handle::cufftHandle,
                                            input_size::NTuple{N,Int}, output_size::NTuple{N,Int},
                                            region::NTuple{R,Int}, buffer::B
                                            ) where {T<:cufftNumber,S<:cufftNumber,K,inplace,N,R,B}
        abs(K) == 1 || throw(ArgumentError("FFT direction must be either -1 (forward) or +1 (inverse)"))
        inplace isa Bool || throw(ArgumentError("FFT inplace argument must be a Bool"))
        p = new{T,S,K,inplace,N,R,B}(handle, context(), stream(), input_size, output_size, region, buffer)
        finalizer(unsafe_free!, p)
        p
    end
end

function CuFFTPlan{T,S,K,inplace,N,R,B}(handle::cufftHandle, X::DenseCuArray{S,N},
                                        sizey::NTuple{N,Int}, region::NTuple{R,Int}, buffer::B
                                        ) where {T<:cufftNumber,S<:cufftNumber,K,inplace,N,R,B}
    CuFFTPlan{T,S,K,inplace,N,R,B}(handle, size(X), sizey, region, buffer)
end

function CUDACore.unsafe_free!(plan::CuFFTPlan)
    if plan.handle != C_NULL
        context!(plan.ctx; skip_destroyed=true) do
            cufftReleasePlan(plan.handle)
        end
        plan.handle = C_NULL
    end
    if !isnothing(plan.buffer)
        CUDACore.unsafe_free!(plan.buffer)
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

# FFT plans can be user-created on a different task, whose stream might be different from
# the one used in the current task. call this function before every API call that performs
# operations on a stream to ensure the plan is using the correct task-local stream.
@inline function update_stream(plan::CuFFTPlan)
    new_stream = stream()
    if plan.stream != new_stream
        plan.stream = new_stream
        cufftSetStream(plan, new_stream)
    end
    return
end


## plan methods

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::DenseCuArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::DenseCuArray{<:Real}, region) = $pf(complexfloat(x), region)
        $f(x::DenseCuArray{<:Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::DenseCuArray{<:Complex{<:Union{Integer,Rational}}}, region) = $pf(complexfloat(x), region)
    end
end
rfft(x::DenseCuArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(realfloat(x), region)
plan_rfft(x::DenseCuArray{<:Real}, region) = plan_rfft(realfloat(x), region)

function irfft(x::DenseCuArray{<:Union{Real,Integer,Rational}}, d::Integer, region=1:ndims(x))
    irfft(complexfloat(x), d, region)
end

"""
    get_batch_dims(region, sz)

returns the dimensions over which to run internal batching and dimensions used for external (for-loop) batching.
It finds the largest product of consecutive dimensions and uses these as internal batch dimensions.
All other dimensions are external batch dimensions.

    internal_batch_dims, external_batch_dims = get_batch_dims(region, sz)

# Parameters:
- `region`: Tuple of dimensions to transform
- `sz`: size of the array to transform. All dimensions not in `region` are considered as batch dimensions.
        This size Tuple is only used to determine the best set of consecutive dimensions to be used for internal batching.
"""
function get_batch_dims(region, sz)
    internal_batch_dims = ()
    external_batch_dims = ()
    previous_transform_dim = 0
    best_gap_size = 0
    # iterate through the transform dimensions and one extra dim beyond the size to cover the external batch dims
    for t in (region..., length(sz)+1)
        # calculate the product only of consecutively non-transformed sizes
        if (t > previous_transform_dim+1)
            gap_size = prod(sz[(previous_transform_dim+1):(t-1)])
            if (gap_size > best_gap_size)
                best_gap_size = gap_size
                # the previously best dims were not the best. Add them to the external list.
                external_batch_dims = (external_batch_dims..., internal_batch_dims...)
                internal_batch_dims = Tuple((previous_transform_dim+1):(t-1))
            else
                external_batch_dims = (external_batch_dims..., ((previous_transform_dim+1):(t-1))...)
            end
        end
        previous_transform_dim = t
    end
    return internal_batch_dims, external_batch_dims
end

# retrieves the size to allocate even if the external batch dimensions do no transform
get_osz(osz, x) = ntuple((d)->(d>length(osz) ? size(x, d) : osz[d]), ndims(x))

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
        Base.@constprop :aggressive function $f(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
            R = length(region)
            region = NTuple{R,Int}(region)
            $f(X, region)
        end
    end
end

# inplace complex
function plan_fft!(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_FORWARD
    inplace = true
    region = ensure_increasing(ensure_unique(region))

    handle = cufftGetPlan(T, T, size(X), region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

function plan_bfft!(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = true
    region = ensure_increasing(ensure_unique(region))

    handle = cufftGetPlan(T, T, size(X), region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

# out-of-place complex
function plan_fft(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_FORWARD
    inplace = false
    region = ensure_increasing(ensure_unique(region))

    handle = cufftGetPlan(T, T, size(X), region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

function plan_bfft(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = false
    region = ensure_increasing(ensure_unique(region))

    handle = cufftGetPlan(T, T, size(X), region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, size(X), size(X), region, nothing)
end

# out-of-place real-to-complex
Base.@constprop :aggressive function plan_rfft(X::DenseCuArray{T,N}, region) where {T<:cufftReals,N}
    R = length(region)
    region = NTuple{R,Int}(region)
    plan_rfft(X, region)
end

function plan_rfft(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftReals,N,R}
    K = CUFFT_FORWARD
    inplace = false
    region = ensure_strictly_increasing(region)

    handle = cufftGetPlan(complex(T), T, size(X), region)

    xdims = size(X)
    ydims = Base.setindex(xdims, div(xdims[region[1]], 2) + 1, region[1])

    # The buffer is not needed for real-to-complex (`mul!`),
    # but it’s required for complex-to-real (`ldiv!`).
    buffer = CuArray{complex(T)}(undef, ydims...)
    B = typeof(buffer)

    CuFFTPlan{complex(T),T,K,inplace,N,R,B}(handle, size(X), (ydims...,), region, buffer)
end

# out-of-place complex-to-real
Base.@constprop :aggressive function plan_brfft(X::DenseCuArray{T,N}, d::Integer, region) where {T<:cufftComplexes,N}
    R = length(region)
    region = NTuple{R,Int}(region)
    plan_brfft(X, d, region)
end

function plan_brfft(X::DenseCuArray{T,N}, d::Integer, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = false
    region = ensure_strictly_increasing(region)

    xdims = size(X)
    ydims = Base.setindex(xdims, d, region[1])
    handle = cufftGetPlan(real(T), T, ydims, region)

    buffer = CuArray{T}(undef, size(X))
    B = typeof(buffer)

    CuFFTPlan{real(T),T,K,inplace,N,R,B}(handle, size(X), ydims, region, buffer)
end


# FIXME: plan_inv methods allocate needlessly (to provide type parameters)
# Perhaps use FakeArray types to avoid this.

function plan_inv(p::CuFFTPlan{T,S,CUFFT_INVERSE,inplace,N,R,B}
                  ) where {T<:cufftNumber,S<:cufftNumber,inplace,N,R,B}
    handle = cufftGetPlan(S, T, p.output_size, p.region)
    ScaledPlan(CuFFTPlan{S,T,CUFFT_FORWARD,inplace,N,R,B}(handle, p.output_size, p.input_size, p.region, p.buffer),
               normalization(real(T), p.output_size, p.region))
end

function plan_inv(p::CuFFTPlan{T,S,CUFFT_FORWARD,inplace,N,R,B}
                  ) where {T<:cufftNumber,S<:cufftNumber,inplace,N,R,B}
    handle = cufftGetPlan(S, T, p.input_size, p.region)
    ScaledPlan(CuFFTPlan{S,T,CUFFT_INVERSE,inplace,N,R,B}(handle, p.output_size, p.input_size, p.region, p.buffer),
               normalization(real(S), p.input_size, p.region))
end


## plan execution

# NOTE: "in-place complex-to-real FFTs may overwrite arbitrary imaginary input point values
#       [...]. Out-of-place complex-to-real FFT will always overwrite input buffer."
#       see # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055.

function assert_applicable(p::CuFFTPlan{T,S}, X::DenseCuArray{S}) where {T,S}
    (size(X) == p.input_size) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,S,K,inplace}, X::DenseCuArray{S},
                           Y::DenseCuArray{T}) where {T,S,K,inplace}
    assert_applicable(p, X)
    if size(Y) != p.output_size
        throw(ArgumentError("CuFFT plan applied to wrong-size output"))
    elseif inplace != (pointer(X) == pointer(Y))
        throw(ArgumentError(string("CuFFT ",
                                   inplace ? "in-place" : "out-of-place",
                                   " plan applied to ",
                                   inplace ? "out-of-place" : "in-place",
                                   " data")))
    end
end


function unsafe_execute!(plan::CuFFTPlan{T,S,K,inplace}, x::DenseCuArray{S}, y::DenseCuArray{T}) where {T,S,K,inplace}
    update_stream(plan)
    cufftXtExec(plan, x, y, K)
end

# 0-based footprint of one cuFFT execution, in elements: max linear offset
# touched + 1. cuFFT covers all combinations of (region ∪ internal_batch_dims);
# each external batch dim is fixed at the current index.
function plan_footprint(sizes::Dims, region, internal_batch_dims)
    covered = (region..., internal_batch_dims...)
    isempty(covered) && return 1
    max_off = 0
    for d in covered
        stride_d = prod(sizes[1:d-1])
        max_off += (sizes[d] - 1) * stride_d
    end
    return max_off + 1
end

# Out-of-place left-rotation by `shift`: dest[i] = src[mod(i - 1 + shift, n) + 1]
# for i in 1..n. Does not mutate `src`.
function shift_copy!(dest::DenseCuArray, src::DenseCuArray, shift::Integer)
    n = length(src)
    shift = mod(shift, n)
    shift == 0 && return unsafe_copyto!(dest, 1, src, 1, n)
    unsafe_copyto!(dest, 1, src, shift + 1, n - shift)
    unsafe_copyto!(dest, n - shift + 1, src, 1, shift)
    return dest
end

# A version of unsafe_execute! that handles external batch dims (dims outside
# the cuFFT plan's internal batching) by issuing one cuFFT call per external
# index. cuFFT's R2C/C2R APIs require each call's base address to be
# Complex-aligned; in 1-based Julia indices that means the batch's linear
# start (`bs`) must be odd when its side of the transform has Real eltype.
#
# Two strategies are used, picked by which side is Real:
#
#   * R2C (input Real, output Complex): rotate x into a fresh aligned
#     `scratch_x` once and read misaligned batches from it. The user's input
#     is never mutated and the extra cost is O(length(x)) regardless of how
#     many batches are misaligned.
#
#   * C2R (input Complex, output Real): per-misaligned-batch
#     read-modify-write through a small `scratch_y` of size `foot_y`. Other
#     external batches share the same footprint range in y, so we seed
#     scratch_y with y's current contents before each cuFFT call and copy
#     back after, preserving their writes.
function unsafe_execute_external_batches!(p::CuFFTPlan{T,S,K,inplace}, x, y) where {T,S,K,inplace}
    region = p.region
    internal_dims, external_dims = get_batch_dims(region, p.output_size)
    if isempty(external_dims)
        unsafe_execute!(p, x, y)
        return
    end

    prefix_prod_x = (1, cumprod(size(x))...)
    prefix_prod_y = (1, cumprod(size(y))...)
    ext_stride_x = map(d -> prefix_prod_x[d], external_dims)
    ext_stride_y = map(d -> prefix_prod_y[d], external_dims)
    ext_size = map(d -> size(x, d), external_dims)
    ci = CartesianIndices(ext_size)
    foot_x = plan_footprint(size(x), region, internal_dims)
    foot_y = plan_footprint(size(y), region, internal_dims)

    # R2C side: find the first misaligned external batch (if any) and pre-rotate
    # x into scratch_x once. The same shift aligns every other misaligned batch
    # in the same stride orbit.
    to_skip_x = 0
    scratch_x = nothing
    if S <: Real
        for c in ci
            bs = sum(ext_stride_x .* (Tuple(c) .- 1)) + 1
            if iseven(bs); to_skip_x = bs - 1; break; end
        end
        if to_skip_x > 0
            scratch_x = shift_copy!(CuArray{S}(undef, length(x)), x, to_skip_x)
        end
    end

    # C2R side: per-misaligned-batch scratch, allocated lazily.
    scratch_y = nothing

    for c in ci
        bs_x = sum(ext_stride_x .* (Tuple(c) .- 1)) + 1
        bs_y = sum(ext_stride_y .* (Tuple(c) .- 1)) + 1
        misaligned_x = S <: Real && iseven(bs_x)
        misaligned_y = T <: Real && iseven(bs_y)

        vx = misaligned_x ?
            (@view scratch_x[bs_x - to_skip_x : end]) :
            (@view x[bs_x : end])
        vy = if misaligned_y
            scratch_y === nothing && (scratch_y = CuArray{T}(undef, foot_y))
            unsafe_copyto!(scratch_y, 1, y, bs_y, foot_y)
            @view scratch_y[1:foot_y]
        else
            @view y[bs_y : end]
        end

        unsafe_execute!(p, vx, vy)

        if misaligned_y
            unsafe_copyto!(y, bs_y, scratch_y, 1, foot_y)
        end
    end
    return
end

## high-level integrations

function LinearAlgebra.mul!(y::DenseCuArray{T}, p::CuFFTPlan{T,S,K,inplace}, x::DenseCuArray{S}
                           ) where {T,S,K,inplace}
    assert_applicable(p, x, y)
    if !inplace && T<:Real
        # Out-of-place complex-to-real FFT will always overwrite input x.
        # We copy the input x in an auxiliary buffer.
        z = p.buffer
        copyto!(z, x)
    else
        z = x
    end
    unsafe_execute_external_batches!(p, z, y)
    y
end

function Base.:(*)(p::CuFFTPlan{T,S,K,true}, x::DenseCuArray{S}) where {T,S,K}
    assert_applicable(p, x)
    unsafe_execute_external_batches!(p, x, x)
    x
end

function Base.:(*)(p::CuFFTPlan{T,S,K,false}, x::DenseCuArray{S1,M}) where {T,S,K,S1,M}
    if T<:Real
        # Out-of-place complex-to-real FFT will always overwrite input x.
        # We copy the input x in an auxiliary buffer.
        z = p.buffer
        copyto!(z, x)
    else
        if S1 != S
            # Convert to the expected input type.
            z = copy1(S, x)
        else
            z = x
        end
    end
    assert_applicable(p, z)
    y = CuArray{T,M}(undef, p.output_size)
    unsafe_execute_external_batches!(p, z, y)
    y
end

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

function CUDA.unsafe_free!(plan::CuFFTPlan)
    if plan.handle != C_NULL
        context!(plan.ctx; skip_destroyed=true) do
            cufftReleasePlan(plan.handle)
        end
        plan.handle = C_NULL
    end
    if !isnothing(plan.buffer)
        CUDA.unsafe_free!(plan.buffer)
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

# yields the maximal dimensions of the plan, for plans starting at dim 1 or ending at the size vector, 
# this is always the full input size
function plan_max_dims(region, sz) 
    if (region[1] == 1 && (length(region) <=1 || all(diff(collect(region)) .== 1)))
        return length(sz)
    else
        return region[end]
    end
end

# retrieves the size to allocate even if the trailing dimensions do no transform
get_osz(osz, x) = ntuple((d)->(d>length(osz) ? size(x, d) : osz[d]), ndims(x))

# returns a view of the front part of the dimensions of the array up to md dimensions
function front_view(X, md)
    t = ntuple((d)->ifelse(d<=md, Colon(), 1), ndims(X))
    @view X[t...]
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

    md = plan_max_dims(region, size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

function plan_bfft!(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = true

    md = plan_max_dims(region, size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

# out-of-place complex
function plan_fft(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_FORWARD
    inplace = false

    md = plan_max_dims(region,size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    CuFFTPlan{T,T,K,inplace,N,R,Nothing}(handle, X, size(X), region, nothing)
end

function plan_bfft(X::DenseCuArray{T,N}, region::NTuple{R,Int}) where {T<:cufftComplexes,N,R}
    K = CUFFT_INVERSE
    inplace = false

    md = plan_max_dims(region,size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

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

    md = plan_max_dims(region,size(X))
    sizex = size(X)[1:md]

    handle = cufftGetPlan(complex(T), T, sizex, region)

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
    md_osz = plan_max_dims(p.region, p.output_size)
    sz_X = p.output_size[1:md_osz]
    handle = cufftGetPlan(S, T, sz_X, p.region)
    ScaledPlan(CuFFTPlan{S,T,CUFFT_FORWARD,inplace,N,R,B}(handle, p.output_size, p.input_size, p.region, p.buffer),
               normalization(real(T), p.output_size, p.region))
end

function plan_inv(p::CuFFTPlan{T,S,CUFFT_FORWARD,inplace,N,R,B}
                  ) where {T<:cufftNumber,S<:cufftNumber,inplace,N,R,B}
    md_isz = plan_max_dims(p.region, p.input_size)
    sz_Y = p.input_size[1:md_isz]
    handle = cufftGetPlan(S, T, sz_Y, p.region)
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

# a version of unsafe_execute which applies the plan to each element of trailing dimensions not covered by the plan.
# Note that for plans, with trailing non-transform dimensions views are created for each of such elements.
# Such views each have lower dimensions and are then transformed by the lower dimension low-level Cuda plan.
function unsafe_execute_trailing!(p, x, y)
    N = plan_max_dims(p.region, p.output_size)
    M = ndims(x)
    d = p.region[end]
    if M == N  
        unsafe_execute!(p,x,y)
    else
        front_ids = ntuple((dd)->Colon(), d)
        for c in CartesianIndices(size(x)[d+1:end])
            ids = ntuple((dd)->c[dd], M-N)
            vx = @view x[front_ids..., ids...]
            vy = @view y[front_ids..., ids...]
            unsafe_execute!(p,vx,vy)
        end
    end
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
    unsafe_execute_trailing!(p, z, y)
    y
end

function Base.:(*)(p::CuFFTPlan{T,S,K,true}, x::DenseCuArray{S}) where {T,S,K}
    assert_applicable(p, x)
    unsafe_execute_trailing!(p, x, x)
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
    unsafe_execute_trailing!(p, z, y)
    y
end

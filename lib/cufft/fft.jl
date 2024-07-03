# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft, irfft,
    Plan, ScaledPlan

using LinearAlgebra

Base.:(*)(p::Plan{T}, x::DenseCuArray) where {T} = p * copy1(T, x)
Base.:(*)(p::ScaledPlan, x::DenseCuArray) = rmul!(p.p * x, p.scale)

input_type(plan::ScaledPlan) = input_type(plan.p)
output_type(plan::ScaledPlan) = output_type(plan.p)

## plan structure

# K is an integer flag for forward/backward
# also used as an alias for r2c/c2r

# inplace is a boolean flag

abstract type CuFFTPlan{T<:cufftNumber, K, inplace} <: Plan{T} end

# for some reason, cufftHandle is an integer and not a pointer...
Base.convert(::Type{cufftHandle}, p::CuFFTPlan) = p.handle
# we also need to be able to convert CuFFTPlans that have been wrapped in a ScaledPlan
Base.convert(::Type{cufftHandle}, p::ScaledPlan{T,P,N}) where {T,N,P<:CuFFTPlan} = convert(cufftHandle, p.p)

function CUDA.unsafe_free!(plan::CuFFTPlan)
    if plan.handle != C_NULL
        context!(plan.ctx; skip_destroyed=true) do
            cufftReleasePlan(plan.handle)
        end
        plan.handle = C_NULL
    end
end

mutable struct cCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    # handle to Cuda low level plan. Note that this plan sometimes has lower dimensions
    # to handle more transform cases such as individual directions
    handle::cufftHandle 
    ctx::CuContext
    stream::CuStream
    input_size::NTuple{N,Int}   # Julia size of input array
    output_size::NTuple{N,Int}  # Julia size of output array
    region::Any
    pinv::ScaledPlan            # required by AbstractFFT API

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle,
                                       input_size::NTuple{N,Int}, output_size::NTuple{N,Int}, region
                                       ) where {T<:cufftNumber,K,inplace,N}
        abs(K) == 1 || throw(ArgumentError("FFT direction must be either -1 (forward) or +1 (inverse)"))
        inplace isa Bool || throw(ArgumentError("FFT inplace argument must be a Bool"))
        p = new{T,K,inplace,N}(handle, context(), stream(), input_size, output_size, region)
        finalizer(unsafe_free!, p)
        p
    end

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::NTuple{N,Int}, region,
                                       ) where {T<:cufftNumber,K,inplace,N}
        cCuFFTPlan{T,K,inplace,N}(handle, size(X), sizey, region)
    end

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, sizex::NTuple{N,Int},
                                       sizey::NTuple{N,Int}, region, xtype::cufftType,
                                       ) where {T<:cufftNumber,K,inplace,N}
        xtype == _cufftType(T, T) || throw(ArgumentError("FFT type $xtype does not match element type $T"))
        cuFFTPlan{T,K,inplace,N}(handle, sizex, sizey, region)
    end

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::NTuple{N,Int}, region, xtype
                                       ) where {T<:cufftNumber,K,inplace,N}
        cCuFFTPlan{T,K,inplace,N}(handle, size(X), sizey, region, xtype)
    end
end

input_type(::cCuFFTPlan{T}) where {T} = T
output_type(::cCuFFTPlan{T}) where {T} = T

# This is a superset of cCuFFTPlan, should probably combine them
mutable struct rCuFFTPlan{T<:cufftNumber,R<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    input_size::NTuple{N,Int}   # Julia size of input array
    output_size::NTuple{N,Int}  # Julia size of output array
    region::Any
    pinv::ScaledPlan            # required by AbstractFFT API

    function rCuFFTPlan{T,R,K,inplace,N}(handle::cufftHandle,
                                         input_size::NTuple{N,Int}, output_size::NTuple{N,Int}, region
                                         ) where {T<:cufftNumber,R<:cufftNumber,K,inplace,N}
        abs(K) == 1 || throw(ArgumentError("FFT direction must be either -1 (forward) or +1 (inverse)"))
        inplace isa Bool || throw(ArgumentError("FFT inplace argument must be a Bool"))
        p = new{T,R,K,inplace,N}(handle, context(), stream(), input_size, output_size, region)
        finalizer(unsafe_free!, p)
        p
    end

    function rCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::NTuple{N,Int}, region,
                                       ) where {T<:cufftNumber,K,inplace,N}
        rCuFFTPlan{T,K,inplace,N}(handle, size(X), sizey, region)
    end

    function rCuFFTPlan{T,R,K,inplace,N}(handle::cufftHandle, sizex::NTuple{N,Int},
                                         sizey::NTuple{N,Int}, region, xtype::cufftType,
                                         ) where {T<:cufftNumber,R<:cufftNumber,K,inplace,N}
        xtype == _cufftType(T, R) || throw(ArgumentError("FFT type $xtype does not match input type type $T and output type $R"))
        cuFFTPlan{T,K,inplace,N}(handle, sizex, sizey, region)
    end

    function rCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::NTuple{N,Int}, region, xtype
                                       ) where {T<:cufftNumber,K,inplace,N}
        rCuFFTPlan{T,K,inplace,N}(handle, size(X), sizey, region, xtype)
    end
end

input_type(::rCuFFTPlan{T,R}) where {T,R} = T
output_type(::rCuFFTPlan{T,R}) where {T,R} = R

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

function Base.show(io::IO, p::CuFFTPlan{T,K,inplace}) where {T,K,inplace}
    R = output_type(p)
    print(io, "CUFFT ",
          inplace ? "in-place " : "",
          R == T ? "$T " : "$(T)-to-$(R) ",
          K == CUFFT_FORWARD ? "forward " : "backward ",
          "plan for ")
    showfftdims(io, p.input_size, T)
end

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

# inplace complex
function plan_fft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = true
    region = Tuple(region)

    md = plan_max_dims(region, size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region)
end


function plan_bfft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = true
    region = Tuple(region)

    md = plan_max_dims(region, size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region)
end

# out-of-place complex
function plan_fft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = false
    region = Tuple(region)

    md = plan_max_dims(region,size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region)
end

function plan_bfft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    region = Tuple(region)

    md = plan_max_dims(region,size(X))
    sizex = size(X)[1:md]
    handle = cufftGetPlan(T, T, sizex, region)

    cCuFFTPlan{T,K,inplace,N}(handle, size(X), size(X), region)
end

# out-of-place real-to-complex
function plan_rfft(X::DenseCuArray{T,N}, region) where {T<:cufftReals,N}
    K = CUFFT_FORWARD
    inplace = false
    region = Tuple(region)

    md = plan_max_dims(region,size(X))
    # X = front_view(X, md)
    sizex = size(X)[1:md]

    handle = cufftGetPlan(T, Complex{T}, sizex, region)

    ydims = collect(size(X))
    ydims[region[1]] = div(ydims[region[1]],2)+1

    rCuFFTPlan{T,K,inplace,N}(handle, size(X), (ydims...,), region)
end

function plan_brfft(X::DenseCuArray{T,N}, d::Integer, region::Any) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    region = Tuple(region)

    ydims = collect(size(X))
    ydims[region[1]] = d

    handle = cufftGetPlan(T, Complex{T}, (ydims...,), region)

    rCuFFTPlan{T,K,inplace,N}(handle, size(X), (ydims...,), region)
end


# FIXME: plan_inv methods allocate needlessly (to provide type parameters)
# Perhaps use FakeArray types to avoid this.

function plan_inv(p::cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}) where {T,N,inplace}
    md = plan_max_dims(p.region, p.input_size)
    sizex = p.input_size[1:md]
    handle = cufftGetPlan(T, T, sizex, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}(handle,  p.input_size,  p.input_size, p.region),
               normalization(real(T), p.input_size, p.region))
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}) where {T,N,inplace}
    md = plan_max_dims(p.region,p.input_size)
    sizex = p.input_size[1:md]
    handle = cufftGetPlan(T, T, sizex, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}(handle,  p.input_size,  p.input_size, p.region),
               normalization(real(T),  p.input_size, p.region))
end

function plan_inv(p::rCuFFTPlan{T,R,CUFFT_INVERSE,inplace,N}
                  ) where {T<:cufftComplexes,R<:cufftReals,N,inplace}
    T == Complex{R} || throw(ArgumentError("Cannot invert plan with mismatching types"))
    md_osz = plan_max_dims(p.region, p.output_size)
    sz_X = p.output_size[1:md_osz]
    handle = cufftGetPlan(R, T, sz_X, p.region)
    ScaledPlan(rCuFFTPlan{R,T,CUFFT_FORWARD,inplace,N}(handle, p.output_size, p.input_size, p.region),
               normalization(R, p.output_size, p.region))
end

function plan_inv(p::rCuFFTPlan{T,R,CUFFT_FORWARD,inplace,N}
                  ) where {T<:cufftReals,R<:cufftComplexes,N,inplace}
    R == Complex{T} || throw(ArgumentError("Cannot invert plan with mismatching types"))
    md_sz = plan_max_dims(p.region,p.input_size)
    sz_Y = p.input_size[1:md_sz]
    handle = cufftGetPlan(R, T, sz_Y, p.region)
    ScaledPlan(rCuFFTPlan{R,T,CUFFT_INVERSE,inplace,N}(handle, p.output_size, p.input_size, p.region),
               normalization(T, p.input_size, p.region))
end


## plan execution

# NOTE: "in-place complex-to-real FFTs may overwrite arbitrary imaginary input point values
#       [...]. Out-of-place complex-to-real FFT will always overwrite input buffer."
#       see # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055.

function assert_applicable(p::CuFFTPlan{T}, X::DenseCuArray{T}) where {T}
    (size(X) == p.input_size) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,K,inplace}, X::DenseCuArray{T},
                           Y::DenseCuArray) where {T,K,inplace}
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


function unsafe_execute!(plan::cCuFFTPlan{T,K,<:Any,M},
                         x::DenseCuArray{T,N},
                         y::DenseCuArray{T}) where {T,K,M,N}
    update_stream(plan)
    cufftXtExec(plan, x, y, K)
end

function unsafe_execute!(plan::rCuFFTPlan{T,R,K,<:Any,M},
                         x::DenseCuArray{T,N},
                         y::DenseCuArray{R}) where {T,R,K,M,N}
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

function LinearAlgebra.mul!(y::DenseCuArray{Ty}, p::CuFFTPlan{T}, x::DenseCuArray{T}
                           ) where {Ty, T}
    assert_applicable(p,x,y)
    unsafe_execute_trailing!(p,x, y)
    return y
end

function Base.:(*)(p::cCuFFTPlan{T,K,true,N}, x::DenseCuArray{T,M}) where {T,K,N,M}
    assert_applicable(p,x)
    unsafe_execute_trailing!(p,x, x)
    x
end

function Base.:(*)(p::rCuFFTPlan{T,R,CUFFT_FORWARD,false,N}, x::DenseCuArray{T,M}
           ) where {T<:cufftReals,R<:cufftComplexes,N,M}
    assert_applicable(p,x)
    y = CuArray{R,M}(undef, p.output_size)
    unsafe_execute_trailing!(p,x, y)
    y
end

function Base.:(*)(p::rCuFFTPlan{T,R,CUFFT_INVERSE,false,N}, x::DenseCuArray{T,M}
           ) where {T<:cufftComplexes,R<:cufftReals,N,M}
    assert_applicable(p,x)
    y = CuArray{R,M}(undef, p.output_size)
    unsafe_execute_trailing!(p,x, y)
    y
end

function Base.:(*)(p::rCuFFTPlan{T,R,CUFFT_INVERSE,false,N}, x::DenseCuArray{T2,M}
    ) where {T<:cufftComplexes,R<:cufftReals,N,M, T2<:cufftReals}
    x = complex.(x)
    p*x
end

function Base.:(*)(p::cCuFFTPlan{T,K,false,N}, x::DenseCuArray{T,M}) where {T,K,N,M}
    assert_applicable(p,x)
    y = CuArray{T,M}(undef, p.output_size)
    unsafe_execute_trailing!(p,x, y)
    y
end

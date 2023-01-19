# integration with AbstractFFTs.jl

@reexport using AbstractFFTs

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!, plan_ifft,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft,
    Plan, ScaledPlan

using LinearAlgebra

Base.:(*)(p::Plan{T}, x::DenseCuArray) where {T} = p * copy1(T, x)
Base.:(*)(p::ScaledPlan, x::DenseCuArray) = rmul!(p.p * x, p.scale)


## plan structure

# K is an integer flag for forward/backward
# also used as an alias for r2c/c2r

# inplace is a boolean flag

abstract type CuFFTPlan{T<:cufftNumber, K, inplace} <: Plan{T} end

# for some reason, cufftHandle is an integer and not a pointer...
Base.convert(::Type{cufftHandle}, p::CuFFTPlan) = p.handle

function CUDA.unsafe_free!(plan::CuFFTPlan)
    if plan.handle != C_NULL
        context!(plan.ctx; skip_destroyed=true) do
            cufftReleasePlan(plan.handle)
        end
        plan.handle = C_NULL
    end
end

mutable struct cCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function cCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::Tuple, region, xtype
                                      ) where {T<:cufftNumber,K,inplace,N}
        # TODO: enforce consistency of sizey
        p = new(handle, context(), stream(), size(X), sizey, xtype, region)
        finalizer(unsafe_free!, p)
        p
    end
end

mutable struct rCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    xtype::cufftType
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function rCuFFTPlan{T,K,inplace,N}(handle::cufftHandle, X::DenseCuArray{T,N},
                                       sizey::Tuple, region, xtype
                                      ) where {T<:cufftNumber,K,inplace,N}
        # TODO: enforce consistency of sizey
        p = new(handle, context(), stream(), size(X), sizey, xtype, region)
        finalizer(unsafe_free!, p)
        p
    end
end

const xtypenames = Dict{cufftType,String}(CUFFT_R2C => "real-to-complex",
                                          CUFFT_C2R => "complex-to-real",
                                          CUFFT_C2C => "complex",
                                          CUFFT_D2Z => "d.p. real-to-complex",
                                          CUFFT_Z2D => "d.p. complex-to-real",
                                          CUFFT_Z2Z => "d.p. complex")

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
    print(io, inplace ? "CUFFT in-place " : "CUFFT ",
          xtypenames[p.xtype],
          K == CUFFT_FORWARD ? " forward" : " backward",
          " plan for ")
    showfftdims(io, p.sz, T)
end

Base.size(p::CuFFTPlan) = p.sz

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

# region is an iterable subset of dimensions
# spec. an integer, range, tuple, or array

# inplace complex
function plan_fft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = true
    xtype = (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle = cufftGetPlan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region, xtype)
end

function plan_bfft!(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = true
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle = cufftGetPlan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region, xtype)
end

# out-of-place complex
function plan_fft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z
    inplace = false

    handle = cufftGetPlan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region, xtype)
end

function plan_bfft(X::DenseCuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    handle = cufftGetPlan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(handle, X, size(X), region, xtype)
end

# out-of-place real-to-complex
function plan_rfft(X::DenseCuArray{T,N}, region) where {T<:cufftReals,N}
    K = CUFFT_FORWARD
    inplace = false
    xtype =  (T == cufftReal) ? CUFFT_R2C : CUFFT_D2Z

    handle = cufftGetPlan(xtype, size(X), region)

    ydims = collect(size(X))
    ydims[region[1]] = div(ydims[region[1]],2)+1

    rCuFFTPlan{T,K,inplace,N}(handle, X, (ydims...,), region, xtype)
end

function plan_brfft(X::DenseCuArray{T,N}, d::Integer, region::Any) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2R : CUFFT_Z2D
    ydims = collect(size(X))
    ydims[region[1]] = d

    handle = cufftGetPlan(xtype, (ydims...,), region)

    rCuFFTPlan{T,K,inplace,N}(handle, X, (ydims...,), region, xtype)
end

# FIXME: plan_inv methods allocate needlessly (to provide type parameters)
# Perhaps use FakeArray types to avoid this.

function plan_inv(p::cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    handle = cufftGetPlan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}(handle, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    handle = cufftGetPlan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}(handle, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_INVERSE,inplace,N}
                  ) where {T<:cufftComplexes,N,inplace}
    X = CuArray{real(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_C2R ? CUFFT_R2C : CUFFT_D2Z
    handle = cufftGetPlan(xtype, p.osz, p.region)
    ScaledPlan(rCuFFTPlan{real(T),CUFFT_FORWARD,inplace,N}(handle, X, p.sz, p.region, xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_FORWARD,inplace,N}
                  ) where {T<:cufftReals,N,inplace}
    X = CuArray{complex(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_R2C ? CUFFT_C2R : CUFFT_Z2D
    handle = cufftGetPlan(xtype, p.sz, p.region)
    ScaledPlan(rCuFFTPlan{complex(T),CUFFT_INVERSE,inplace,N}(handle, X, p.sz,
                                                              p.region, xtype),
               normalization(Y, p.region))
end


## plan execution

# NOTE: "in-place complex-to-real FFTs may overwrite arbitrary imaginary input point values
#       [...]. Out-of-place complex-to-real FFT will always overwrite input buffer."
#       see # JuliaGPU/CuArrays.jl#345, NVIDIA/cuFFT#2714055.

function assert_applicable(p::CuFFTPlan{T}, X::DenseCuArray{T}) where {T}
    (size(X) == p.sz) ||
        throw(ArgumentError("CuFFT plan applied to wrong-size input"))
end

function assert_applicable(p::CuFFTPlan{T,K,inplace}, X::DenseCuArray{T},
                           Y::DenseCuArray) where {T,K,inplace}
    assert_applicable(p, X)
    if size(Y) != p.osz
        throw(ArgumentError("CuFFT plan applied to wrong-size output"))
    elseif inplace != (pointer(X) == pointer(Y))
        throw(ArgumentError(string("CuFFT ",
                                   inplace ? "in-place" : "out-of-place",
                                   " plan applied to ",
                                   inplace ? "out-of-place" : "in-place",
                                   " data")))
    end
end

function unsafe_execute!(plan::cCuFFTPlan{cufftComplex,K,<:Any,N},
                         x::DenseCuArray{cufftComplex,N},
                         y::DenseCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2C
    update_stream(plan)
    cufftExecC2C(plan, x, y, K)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,true,N},
                         x::DenseCuArray{cufftComplex,N},
                         y::DenseCuArray{cufftReal,N}) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    update_stream(plan)
    cufftExecC2R(plan, x, y)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftComplex,K,false,N},
                         x::DenseCuArray{cufftComplex,N},
                         y::DenseCuArray{cufftReal}) where {K,N}
    @assert plan.xtype == CUFFT_C2R
    x = copy(x)
    update_stream(plan)
    cufftExecC2R(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftReal,K,<:Any,N},
                         x::DenseCuArray{cufftReal,N},
                         y::DenseCuArray{cufftComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_R2C
    update_stream(plan)
    cufftExecR2C(plan, x, y)
end

function unsafe_execute!(plan::cCuFFTPlan{cufftDoubleComplex,K,<:Any,N},
                         x::DenseCuArray{cufftDoubleComplex,N},
                         y::DenseCuArray{cufftDoubleComplex}) where {K,N}
    @assert plan.xtype == CUFFT_Z2Z
    update_stream(plan)
    cufftExecZ2Z(plan, x, y, K)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,true,N},
                         x::DenseCuArray{cufftDoubleComplex,N},
                         y::DenseCuArray{cufftDoubleReal}) where {K,N}
    update_stream(plan)
    @assert plan.xtype == CUFFT_Z2D
    cufftExecZ2D(plan, x, y)
end
function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleComplex,K,false,N},
                         x::DenseCuArray{cufftDoubleComplex,N},
                         y::DenseCuArray{cufftDoubleReal}) where {K,N}
    @assert plan.xtype == CUFFT_Z2D
    x = copy(x)
    update_stream(plan)
    cufftExecZ2D(plan, x, y)
    unsafe_free!(x)
end

function unsafe_execute!(plan::rCuFFTPlan{cufftDoubleReal,K,<:Any,N},
                         x::DenseCuArray{cufftDoubleReal,N},
                         y::DenseCuArray{cufftDoubleComplex,N}) where {K,N}
    @assert plan.xtype == CUFFT_D2Z
    update_stream(plan)
    cufftExecD2Z(plan, x, y)
end


## high-level integrations

function LinearAlgebra.mul!(y::DenseCuArray{Ty}, p::CuFFTPlan{T}, x::DenseCuArray{T}
                           ) where {Ty, T}
    assert_applicable(p,x,y)
    unsafe_execute!(p,x,y)
    return y
end

function Base.:(*)(p::cCuFFTPlan{T,K,true,N}, x::DenseCuArray{T,N}) where {T,K,N}
    assert_applicable(p,x)
    unsafe_execute!(p,x,x)
    x
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_FORWARD,false,N}, x::DenseCuArray{T,N}
           ) where {T<:cufftReals,N}
    assert_applicable(p,x)
    @assert p.xtype ∈ [CUFFT_R2C,CUFFT_D2Z]
    y = CuArray{complex(T),N}(undef, p.osz)
    unsafe_execute!(p,x,y)
    y
end

function Base.:(*)(p::rCuFFTPlan{T,CUFFT_INVERSE,false,N}, x::DenseCuArray{T,N}
           ) where {T<:cufftComplexes,N}
    assert_applicable(p,x)
    @assert p.xtype ∈ [CUFFT_C2R,CUFFT_Z2D]
    y = CuArray{real(T),N}(undef, p.osz)
    unsafe_execute!(p,x,y)
    y
end

function Base.:(*)(p::cCuFFTPlan{T,K,false,N}, x::DenseCuArray{T,N}) where {T,K,N}
    assert_applicable(p,x)
    y = CuArray{T,N}(undef, p.osz)
    unsafe_execute!(p,x,y)
    y
end

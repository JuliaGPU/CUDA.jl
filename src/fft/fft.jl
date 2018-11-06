# K is a flag for forward/backward
# also used as an alias for r2c/c2r

abstract type CuFFTPlan{T<:cufftNumber, K, inplace} <: Plan{T} end

mutable struct cCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    plan::cufftHandle_t
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    xtype::Int
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function cCuFFTPlan{T,K,inplace,N}(plan::cufftHandle_t, X::CuArray{T,N},
                                       sizey::Tuple, region, xtype::Integer
                                       ) where {T<:cufftNumber,K,inplace,N}
        # maybe enforce consistency of sizey
        p = new(plan, size(X), sizey, xtype, region)
        finalizer(destroy_plan, p)
        p
    end
end

cCuFFTPlan(plan,X,region,xtype::Integer) = cCuFFTPlan(plan,X,size(X),region,xtype)

mutable struct rCuFFTPlan{T<:cufftNumber,K,inplace,N} <: CuFFTPlan{T,K,inplace}
    plan::cufftHandle_t
    sz::NTuple{N,Int} # Julia size of input array
    osz::NTuple{N,Int} # Julia size of output array
    xtype::Int
    region::Any
    pinv::ScaledPlan # required by AbstractFFT API

    function rCuFFTPlan{T,K,inplace,N}(plan::cufftHandle_t, X::CuArray{T,N},
                                       sizey::Tuple, region, xtype::Integer
                                       ) where {T<:cufftNumber,K,inplace,N}
        # maybe enforce consistency of sizey
        p = new(plan, size(X), sizey, xtype, region)
        finalizer(destroy_plan, p)
        p
    end
end

rCuFFTPlan(plan,X,region,xtype::Integer) = rCuFFTPlan(plan,X,size(X),region,xtype)

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

function show(io::IO, p::CuFFTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "CUFFT in-place " : "CUFFT ",
          xtypenames[p.xtype],
          K == CUFFT_FORWARD ? " forward" : " backward",
          " plan for ")
    showfftdims(io, p.sz, T)
end

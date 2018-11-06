# region is an iterable subset of dimensions
# spec. an integer, range, tuple, or array

# inplace complex
function plan_fft!(X::CuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    inplace = true
    xtype = (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    pp = _mkplan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(pp, X, size(X), region, xtype)
end

function plan_bfft!(X::CuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = true
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    pp = _mkplan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(pp, X, size(X), region, xtype)
end

# out-of-place complex
function plan_fft(X::CuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_FORWARD
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z
    inplace = false

    pp = _mkplan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(pp, X, size(X), region, xtype)
end

function plan_bfft(X::CuArray{T,N}, region) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2C : CUFFT_Z2Z

    pp = _mkplan(xtype, size(X), region)

    cCuFFTPlan{T,K,inplace,N}(pp, X, size(X), region, xtype)
end

# out-of-place real-to-complex
function plan_rfft(X::CuArray{T,N}, region) where {T<:cufftReals,N}
    K = CUFFT_FORWARD
    inplace = false
    xtype =  (T == cufftReal) ? CUFFT_R2C : CUFFT_D2Z

    pp = _mkplan(xtype, size(X), region)

    ydims = collect(size(X))
    ydims[region[1]] = div(ydims[region[1]],2)+1

    rCuFFTPlan{T,K,inplace,N}(pp, X, (ydims...,), region, xtype)
end

function plan_brfft(X::CuArray{T,N}, d::Integer, region::Any) where {T<:cufftComplexes,N}
    K = CUFFT_INVERSE
    inplace = false
    xtype =  (T == cufftComplex) ? CUFFT_C2R : CUFFT_Z2D
    ydims = collect(size(X))
    ydims[region[1]] = d

    pp = _mkplan(xtype, (ydims...,), region)

    rCuFFTPlan{T,K,inplace,N}(pp, X, (ydims...,), region, xtype)
end

# FIXME: plan_inv methods allocate needlessly (to provide type parameters)
# Perhaps use FakeArray types to avoid this.

function plan_inv(p::cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    pp = _mkplan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}(pp, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::cCuFFTPlan{T,CUFFT_INVERSE,inplace,N}) where {T,N,inplace}
    X = CuArray{T}(undef, p.sz)
    pp = _mkplan(p.xtype, p.sz, p.region)
    ScaledPlan(cCuFFTPlan{T,CUFFT_FORWARD,inplace,N}(pp, X, p.sz, p.region,
                                                     p.xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_INVERSE,inplace,N}
                  ) where {T<:cufftComplexes,N,inplace}
    X = CuArray{real(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_C2R ? CUFFT_R2C : CUFFT_D2Z
    pp = _mkplan(xtype, p.osz, p.region)
    ScaledPlan(rCuFFTPlan{real(T),CUFFT_FORWARD,inplace,N}(pp, X, p.sz, p.region,
                                                     xtype),
               normalization(X, p.region))
end

function plan_inv(p::rCuFFTPlan{T,CUFFT_FORWARD,inplace,N}
                  ) where {T<:cufftReals,N,inplace}
    X = CuArray{complex(T)}(undef, p.osz)
    Y = CuArray{T}(undef, p.sz)
    xtype = p.xtype == CUFFT_R2C ? CUFFT_C2R : CUFFT_Z2D
    pp = _mkplan(xtype, p.sz, p.region)
    ScaledPlan(rCuFFTPlan{complex(T),CUFFT_INVERSE,inplace,N}(pp, X, p.sz,
                                                              p.region, xtype),
               normalization(Y, p.region))
end


# The rest of the standard API

size(p::CuFFTPlan) = p.sz

function mul!(y::CuArray{Ty}, p::CuFFTPlan{T,K,false}, x::CuArray{T}
                  ) where {Ty,T,K}
    assert_applicable(p,x,y)
    unsafe_execute!(p,x,y)
    return y
end

function *(p::cCuFFTPlan{T,K,true,N}, x::CuArray{T,N}) where {T,K,N}
    assert_applicable(p,x)
    unsafe_execute!(p,x)
    x
end

function *(p::rCuFFTPlan{T,CUFFT_FORWARD,false,N}, x::CuArray{T,N}
           ) where {T<:cufftReals,N}
    @assert p.xtype ∈ [CUFFT_R2C,CUFFT_D2Z]
    y = CuArray{complex(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function *(p::rCuFFTPlan{T,CUFFT_INVERSE,false,N}, x::CuArray{T,N}
           ) where {T<:cufftComplexes,N}
    @assert p.xtype ∈ [CUFFT_C2R,CUFFT_Z2D]
    y = CuArray{real(T),N}(undef, p.osz)
    mul!(y,p,x)
    y
end

function *(p::cCuFFTPlan{T,K,false,N}, x::CuArray{T,N}) where {T,K,N}
    y = CuArray{T,N}(undef, p.osz)
    mul!(y,p,x)
    y
end
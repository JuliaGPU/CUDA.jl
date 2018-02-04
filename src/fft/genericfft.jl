cufftfloat(x) = _cufftfloat(float(x))
_cufftfloat(::Type{T}) where {T<:cufftReals} = T
_cufftfloat(::Type{Float16}) = Float32
_cufftfloat(::Type{Complex{T}}) where {T} = Complex{_cufftfloat(T)}
_cufftfloat(::Type{T}) where {T} = error("type $T not supported")
_cufftfloat(x::T) where {T} = _cufftfloat(T)(x)

complexfloat(x::CuArray{Complex{<:cufftReals}}) = x
realfloat(x::CuArray{<:cufftReals}) = x

complexfloat(x::CuArray{T}) where {T<:Complex} = copy1(typeof(cufftfloat(zero(T))), x)
complexfloat(x::CuArray{T}) where {T<:Real} = copy1(typeof(complex(cufftfloat(zero(T)))), x)

realfloat(x::CuArray{T}) where {T<:Real} = copy1(typeof(cufftfloat(zero(T))), x)

function copy1(::Type{T}, x) where T
    y = CuArray{T}(map(length, indices(x)))
    #copy!(y, x)
    y .= broadcast(xi->convert(T,xi),x)
end

# promote to a complex floating-point type (out-of-place only),
# so implementations only need Complex{Float} methods
for f in (:fft, :bfft, :ifft)
    pf = Symbol("plan_", f)
    @eval begin
        $f(x::CuArray{<:Real}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::CuArray{<:Real}, region) = $pf(complexfloat(x), region)
        $f(x::CuArray{<:Complex{<:Union{Integer,Rational}}}, region=1:ndims(x)) = $f(complexfloat(x), region)
        $pf(x::CuArray{<:Complex{<:Union{Integer,Rational}}}, region) = $pf(complexfloat(x), region)
    end
end
rfft(x::CuArray{<:Union{Integer,Rational}}, region=1:ndims(x)) = rfft(realfloat(x), region)
plan_rfft(x::CuArray{<:Real}, region) = plan_rfft(realfloat(x), region)

*(p::Plan{T}, x::CuArray) where {T} = p * copy1(T, x)
*(p::ScaledPlan, x::CuArray) = scale!(p.p * x, p.scale)

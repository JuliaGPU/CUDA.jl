using Statistics

Statistics.varm(A::CuArray{<:Real},m::AbstractArray{<:Real}; dims, corrected::Bool=true) =
    sum((A .- m).^2, dims=dims)/(prod(size(A)[[dims...]])::Int-corrected)

Statistics.stdm(A::CuArray{<:Real},m::AbstractArray{<:Real}, dim::Int; corrected::Bool=true) =
    sqrt.(varm(A,m;dims=dim,corrected=corrected))

Statistics._std(A::CuArray, corrected::Bool, mean, dims) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=dims))

Statistics._std(A::CuArray, corrected::Bool, mean, ::Colon) =
    Base.sqrt.(Statistics.var(A; corrected=corrected, mean=mean, dims=:))

# Revert https://github.com/JuliaLang/Statistics.jl/pull/25
Statistics._mean(A::CuArray, ::Colon)    = sum(A) / length(A)
Statistics._mean(f, A::CuArray, ::Colon) = sum(f, A) / length(A)
Statistics._mean(A::CuArray, dims)    = mean!(Base.reducedim_init(t -> t/2, +, A, dims), A)
Statistics._mean(f, A::CuArray, dims) = sum(f, A, dims=dims) / mapreduce(i -> size(A, i), *, unique(dims); init=1)

function Statistics.covzm(x::CuMatrix, vardim::Int=1; corrected::Bool=true)
    C = Statistics.unscaled_covzm(x, vardim)
    T = promote_type(typeof(one(eltype(C)) / 1), eltype(C))
    A = convert(AbstractArray{T}, C)
    b = 1//(size(x, vardim) - corrected)
    A .*= b
    return A
end

function Statistics.cov2cor!(C::CuMatrix{T}, xsd::CuArray) where T
    nx = length(xsd)
    size(C) == (nx, nx) || throw(DimensionMismatch("inconsistent dimensions"))
    tril!(C, -1)
    C += adjoint(C)
    C = Statistics.clampcor.(C ./ (xsd * xsd'))
    C[diagind(C)] .= oneunit(T)
    return C
end

function Statistics.corzm(x::CuMatrix, vardim::Int=1)
    c = Statistics.unscaled_covzm(x, vardim)
    return Statistics.cov2cor!(c, sqrt.(diag(c)))
end

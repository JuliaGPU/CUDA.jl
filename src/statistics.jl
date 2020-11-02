using Statistics
using Base: require_one_based_indexing

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
    A .= A .* b
    return A
end

function Statistics.cov2cor!(C::CuMatrix{T}, xsd::CuArray) where T
    require_one_based_indexing(C, xsd)
    nx = length(xsd)
    size(C) == (nx, nx) || throw(DimensionMismatch("inconsistent dimensions"))
    tril!(C, -1)
    C += adjoint(C)
    C = Statistics.clampcor.(C ./ (xsd * xsd'))
    C[diagind(C)] .= oneunit(T)
    return C
end

function sqrt!(ret::CuDeviceMatrix{Complex{T}}, z::CuDeviceMatrix{Complex{T}}) where T
    (lx, ly) = size(z)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for j = index_y:stride_y:ly
        for i = index_x:stride_x:lx
            rz = real(z[i, j])
            iz = imag(z[i, j])
            r = sqrt((hypot(rz, iz) + abs(rz)) / 2.0)
            if r == 0
                ret[i, j] = iz * im
            elseif rz >= 0
                ret[i, j] = r + (iz / r / 2.0) * im
            else
                ret[i, j] = (abs(iz) / r / 2.0) + copysign(r, iz)
            end
        end
    end
end

function sqrt!(ret::CuDeviceVector{Complex{T}}, z::CuDeviceVector{Complex{T}}) where T
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(z)
        rz = real(z[i])
        iz = imag(z[i])
        r = sqrt((hypot(rz, iz) + abs(rz)) / 2.0)
        if r == 0
            ret[i] = iz * im
        elseif rz >= 0
            ret[i] = r + (iz / r / 2.0) * im
        else
            ret[i] = (abs(iz) / r / 2.0) + copysign(r, iz)
        end
    end
end

function sqrt(z::Union{CuVector{Complex{T}}, CuMatrix{Complex{T}}}) where T
    function get_config(kernel)
        fun = kernel.fun
        config = launch_configuration(fun)
        return (threads=config.threads, blocks=cld(length(z), config.threads))
    end
    ret = similar(z)
    @cuda config=get_config sqrt!(ret, z)
    ret
end

function sqrt(A::CuArray)
    sqrt.(A)
end

function Statistics.corzm(x::CuMatrix, vardim::Int=1)
    c = Statistics.unscaled_covzm(x, vardim)
    return Statistics.cov2cor!(c, sqrt(diag(c)))
end

using Base.Cartesian

@generated function index_kernel(dest::AbstractArray, src::AbstractArray, idims, Is)
    N = length(Is.parameters)
    quote
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        is = ind2sub(idims, i)
        @nexprs $N i -> @inbounds I_i = Is[i][is[i]]
        @inbounds dest[i] = @ncall $N getindex src i -> I_i
        return
    end
end

function Base._unsafe_getindex!(dest::CuArray, src::CuArray, Is::Union{Real, AbstractArray}...)
    idims = map(length, Is)
    @cuda (1, length(dest)) index_kernel(todevice(dest), todevice(src), idims, Is)
    return dest
end

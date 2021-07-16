import Base.kron

export kron, kron!

function kron_3D!(A::CuDeviceArray{<:Number},B::CuDeviceArray{<:Number},C::CuDeviceArray{<:Number})

    C_rows = size(C,1)
    B_rows = size(B,1)
    B_cols = size(B,2)

    index1 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride1 = blockDim().x * gridDim().x

    index2 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride2 = blockDim().y * gridDim().y

    index3 = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    stride3 = blockDim().z * gridDim().z

    @inbounds for j = index1:stride1:size(A,2), l = index2:stride2:B_cols, i = index3:stride3:size(A,1)
        aij = A[i,j]
        @inbounds for k = 1:B_rows
            C[(B_rows*(i-1) + k) + C_rows*((B_cols*(j-1) + l)-1)] = aij*B[k,l]
        end
    end
end

function kron!(A::CuDeviceArray{<:Number},B::CuDeviceArray{<:Number},C::CuDeviceArray{<:Number})

    p = size(B,1)
    q = size(B,2)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x

    @inbounds for element=index:stride:length(C)
        j = ceil(Int,(element/size(C,1)))
        i = ((element-1)%size(C,1))+1
        C[element] = A[ceil(Int,(i/p)),ceil(Int,j/q)]*B[((i-1)%p)+1, ((j-1)%q)+1]
    end
end


function kron(A::CuArray{S},B::CuArray{T}) where {S,T <: Number}

    col = size(A,2)*size(B,2)
    if col == one(col)
        C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1))
    else
        C = CUDA.zeros(promote_type(S,T),size(A,1)*size(B,1),col)
    end

    dev = device()
    wanted_threads = nextwarp(dev, length(C))
    kron_kernel = @cuda launch=false kron!(A,B,C)
    kernel_config = launch_configuration(kron_kernel.fun)
    max_threads = kernel_config.threads
    nthreads = wanted_threads > max_threads ? prevwarp(dev,max_threads) : wanted_threads
    nblocks=cld(length(C), nthreads)
    kron_kernel(A,B,C,threads=nthreads,blocks=nblocks)
    return C
end

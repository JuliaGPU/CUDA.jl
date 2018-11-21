# Work-inefficient inclusive scan
# - uses shared memory to reduce
#
# Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

using CUDAdrv, CUDAnative, CuArrays

function cpu_accumulate!(op::Function, data::Matrix{T}) where {T}
    cols = size(data,2)
    for col in 1:cols
        accum = zero(T)
        rows = size(data,1)
        for row in 1:size(data,1)
            accum = op(accum, data[row,col])
            data[row,col] = accum
        end
    end
end

function gpu_accumulate!(op::Function, data::CuDeviceMatrix{T}) where {T}
    col = blockIdx().x
    cols = gridDim().x

    row = threadIdx().x
    rows = blockDim().x

    if col <= cols && row <= rows
        shmem = @cuDynamicSharedMem(T, 2*rows)
        shmem[row] = data[row,col]
        sync_threads()

        # parallel reduction
        pin, pout = 1, 0
        offset = 1
        while offset < rows
            pout = 1 - pout
            pin = 1 - pin
            if row > offset
                shmem[pout * rows + row] =
                    op(shmem[pin * rows + row],
                       shmem[pin * rows + row - offset])
            else
                 shmem[pout * rows + row] =
                    shmem[pin * rows + row]
            end
            sync_threads()
            offset *= UInt32(2)
        end
        shmem[pin * rows + row] = shmem[pout * rows + row]
        sync_threads()

        # write back results
        data[row,col] = shmem[row]
    end

    return
end

rows = 5
cols = 4

a = rand(Int, rows, cols)

cpu_a = copy(a)
cpu_accumulate!(+, cpu_a)

gpu_a = CuArray(a)
@cuda blocks=cols threads=rows shmem=2*rows*sizeof(eltype(a)) gpu_accumulate!(+, gpu_a)

using Test

@test cpu_a ≈ Array(gpu_a)


# FURTHER IMPROVEMENTS:
# - work efficiency
# - avoid memory bank conflcits
# - large array support

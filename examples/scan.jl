# Work-inefficient inclusive scan
# - uses shared memory to reduce
#
# Based on http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html

using CUDAdrv, CUDAnative
using Base.Test

function cpu_accumulate!{F<:Function,T}(op::F, data::Matrix{T})
    cols = size(data,2)
    for col in 1:cols
        accum = zero(T)
        rows = size(data,1)
        for row in 1:size(data,1)
            accum = op(accum, data[row,col])
            data[row,col] = accum
        end
    end

    return
end

function gpu_accumulate!{F<:Function,T}(op::F, data::CuDeviceArray{T,2})
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

dev = CuDevice(0)
ctx = CuContext(dev)

rows = 5
cols = 4

a = ones(rows, cols)
a = collect(reshape(1:rows*cols, (rows,cols)))

cpu_a = copy(a)
cpu_accumulate!(+, cpu_a)

gpu_a = CuArray(a)
@cuda (cols,rows, cols*rows*sizeof(eltype(a))) gpu_accumulate!(+, gpu_a)

@test cpu_a ≈ Array(gpu_a)

destroy(ctx)


# FURTHER IMPROVEMENTS:
# - work efficiency
# - avoid memory bank conflcits
# - large array support

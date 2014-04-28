# CUDA implementation of abstractions
module MyCudaModule

using CUDA

export cuda_scan, cuda_map, cuda_reduce

function cuda_scan(data, num_rows, num_cols, results)
    temp = cuSharedMem()
    row = threadId_x()
    col = blockId_x()
    setCuSharedMem(temp, row, data[row + (col-1)*num_rows])
    sync_threads()

    # calculate scan
    pin, pout = 1, 0
    offset = 1
    while offset < num_rows
        pout = 1 - pout
        pin = 1 - pin
        if row > offset
            a = getCuSharedMem(temp, pin * num_rows + row)
            b = getCuSharedMem(temp, pin * num_rows + row - offset)
            setCuSharedMem(temp, pout * num_rows + row, a+b)
        else
            c = getCuSharedMem(temp, pin * num_rows + row)
            setCuSharedMem(temp, pout * num_rows + row, c+0)
        end
        sync_threads()
        offset = offset * 2
    end
    x = getCuSharedMem(temp, pout * num_rows + row)
    setCuSharedMem(temp, pin * num_rows + row, x+0)

    # write back results
    results[row + (col-1)*num_rows] = getCuSharedMem(temp, num_rows + row)

    return nothing
end

function cuda_map(data, num_rows, num_cols, results)
    temp = cuSharedMem()
    row = threadId_x()
    col = blockId_x()
    setCuSharedMem(temp, row, data[row + (col-1)*num_rows])
    sync_threads()

    c = getCuSharedMem(temp, row)
    setCuSharedMem(temp, row, 2*c)

    sync_threads()
    results[row + (col-1)*num_rows] = getCuSharedMem(temp, row)

    return nothing
end

function cuda_reduce(data, num_rows, num_cols, results)
    temp = cuSharedMem()
    row = threadId_x()
    col = blockId_x()
    setCuSharedMem(temp, row, data[row + (col-1)*num_rows])
    sync_threads()

    # calculate scan
    offset = 1
    while offset < num_rows
        if row > offset
            a = getCuSharedMem(temp, row)
            b = getCuSharedMem(temp, row - offset)
            setCuSharedMem(temp, row, a+b)
        end
        sync_threads()
        offset = offset * 2
    end

    # write back results
    if row == num_rows
        results[col] = getCuSharedMem(temp, row)
    end

    return nothing
end

end



# High level abstractions (2 dims)
module Abstractions

using CUDA
using MyCudaModule

export ROWS, COLS, scan, map, reduce

ROWS = 1
COLS = 2

# select a CUDA device
dev = CuDevice(0)

# create a context (like a process in CPU) on the selected device
ctx = create_context(dev)

# finalize: unload module and destroy context -> when to do this?
# unload(md)
# destroy(ctx)

function scan(data, dimension, operation)
    rows, cols = size(data)
    results = Array(eltype(data),size(data))
    grid_size = (dimension == ROWS) ? (rows,1,1) : (cols,1,1) # blocks per grid
    block_size = (dimension == ROWS) ? (cols,1,1): (rows,1,1) # threads per block
    shmem = (dimension == ROWS) ? cols*sizeof(eltype(data)) : rows*sizeof(eltype(data)) # shared memory

    @cuda (MyCudaModule, grid_size, block_size, 2*shmem) cuda_scan(CuIn(data), rows, cols, CuOut(results))
    
    return results
end

function map(data, operation)
    rows, cols = size(data)
    results = Array(eltype(data),size(data))

    grid_size = (cols,1,1) # blocks per grid
    block_size = (rows,1,1) # threads per block
    shmem = rows*sizeof(eltype(data)) # shared memory

    @cuda (MyCudaModule, grid_size, block_size, shmem) cuda_map(CuIn(data), rows, cols, CuOut(results))

    return results
end

function reduce(data, dimension, operation)
    rows, cols = size(data)
    results = Array(eltype(data),size(data, dimension))

    grid_size = (dimension == ROWS) ? (rows,1,1) : (cols,1,1) # blocks per grid
    block_size = (dimension == ROWS) ? (cols,1,1): (rows,1,1) # threads per block
    shmem = (dimension == ROWS) ? cols*sizeof(eltype(data)) : rows*sizeof(eltype(data)) # shared memory

    @cuda (MyCudaModule, grid_size, block_size, shmem) cuda_reduce(CuIn(data), rows, cols, CuOut(results))

    return results
    return nothing
end

end



# Testing
using Abstractions
image = ones(Int64, (10,10))
#results = scan(image, COLS, +)
results = reduce(image, COLS, +)

println("image:")
println(image)
println("results:")
println(results)
# CUDA implementation of abstractions
module __ptx__Abstractions

using CUDA

export cuda_scan, cuda_map, cuda_reduce, cuda_rotate
export SUM, SQUARE

# for scan/reduce
SUM = 0

# for map
SQUARE = 0

function cuda_scan(data, num_rows, num_cols, results, operation)
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
            if operation == 0
                setCuSharedMem(temp, pout * num_rows + row, a+b)
            end
        else
            c = getCuSharedMem(temp, pin * num_rows + row)
            setCuSharedMem(temp, pout * num_rows + row, c+0)
        end
        sync_threads()
        offset = offset * 2
    end
    x = getCuSharedMem(temp, pout * num_rows + row)
    setCuSharedMem(temp, pin * num_rows + row, x+0)
    sync_threads()

    # write back results
    results[row + (col-1)*num_rows] = getCuSharedMem(temp, num_rows + row)

    return nothing
end

function cuda_map(data, num_rows, num_cols, results, operation)
    row = threadId_x()
    col = blockId_x()

    if operation == 0
        results[row + (col-1)*num_rows] = 2*data[row + (col-1)*num_rows]
    end

    return nothing
end

function cuda_reduce(data, num_rows, num_cols, results, operation)
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
            if operation == 0
                setCuSharedMem(temp, pout * num_rows + row, a+b)
            end
        else
            c = getCuSharedMem(temp, pin * num_rows + row)
            setCuSharedMem(temp, pout * num_rows + row, c+0)
        end
        sync_threads()
        offset = offset * 2
    end
    x = getCuSharedMem(temp, pout * num_rows + row)
    setCuSharedMem(temp, pin * num_rows + row, x+0)
    sync_threads()

    # write back results
    if row == num_rows
        results[col] = getCuSharedMem(temp, row)
    end

    return nothing
end

function cuda_rotate(data, angle, result)
    # Compute the thread dimensions
    col  = blockId_x()
    cols = numBlocks_x()
    row  = threadId_x()
    rows = numThreads_x()

    # Calculate transform matrix values
    angle_cos = CUDA.cos(angle)
    angle_sin = CUDA.sin(angle)

    # Calculate origin
    xo = (cols+1) / 2.0
    yo = (rows+1) / 2.0

    # Calculate the source location
    xt = col - xo
    yt = row - yo
    x =  xt * angle_cos + yt * angle_sin + xo
    y = -xt * angle_sin + yt * angle_cos + yo

    # Get fractional and integral part of the coordinates
    x_int = CUDA.floor(x)
    y_int = CUDA.floor(y)
    x_fract = x - x_int
    y_fract = y - y_int

    if (x >= 1 && x < cols && y >= 1 && y < rows)
        result[row + (col-1) * rows] = data[y_int +     (x_int - 1) * rows] * (1 - x_fract) * (1 - y_fract) +
                                       data[y_int +      x_int      * rows] *  x_fract      * (1 - y_fract) +
                                       data[y_int + 1 + (x_int - 1) * rows] * (1 - x_fract) *  y_fract      +
                                       data[y_int + 1 +  x_int      * rows] *  x_fract      *  y_fract
    elseif (col <= cols && row <= rows)
        result[row + (col-1) * rows] = 0
    end

    return nothing
end

end



# High level abstractions (2 dims)
module Abstractions

using CUDA
using __ptx__Abstractions

export ROWS, COLS, SUM, SQUARE
export scan, map, reduce, rotate

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
    results = CuArray(eltype(data),size(data))

    grid_size = (dimension == ROWS) ? (rows,1,1) : (cols,1,1) # blocks per grid
    block_size = (dimension == ROWS) ? (cols,1,1): (rows,1,1) # threads per block
    shmem = (dimension == ROWS) ? cols*sizeof(eltype(data)) : rows*sizeof(eltype(data)) # shared memory

    @cuda (__ptx__Abstractions, grid_size, block_size, 2*shmem) cuda_scan(CuIn(data), rows, cols, CuOut(results), operation)
    
    return results
end

function map(data, operation)
    rows, cols = size(data)
    results = CuArray(eltype(data),size(data))

    grid_size = (cols,1,1) # blocks per grid
    block_size = (rows,1,1) # threads per block
    shmem = rows*sizeof(eltype(data)) # shared memory

    @cuda (__ptx__Abstractions, grid_size, block_size, shmem) cuda_map(CuIn(data), rows, cols, CuOut(results), operation)

    return results
end

function reduce(data, dimension, operation)
    rows, cols = size(data)
    results = CuArray(eltype(data),size(data, dimension))

    grid_size = (dimension == ROWS) ? (rows,1,1) : (cols,1,1) # blocks per grid
    block_size = (dimension == ROWS) ? (cols,1,1): (rows,1,1) # threads per block
    shmem = (dimension == ROWS) ? cols*sizeof(eltype(data)) : rows*sizeof(eltype(data)) # shared memory

    @cuda (__ptx__Abstractions, grid_size, block_size, 2*shmem) cuda_reduce(CuIn(data), rows, cols, CuOut(results), operation)

    return results
end

# angle in radialen
function rotate(data, angle)
    rows, cols = size(data)
    results = CuArray(eltype(data),size(data))

    grid_size = (cols,1,1) # blocks per grid
    block_size = (rows,1,1) # threads per block
    #shmem = rows*sizeof(eltype(data)) # shared memory

    @cuda (__ptx__Abstractions, grid_size, block_size) cuda_rotate(CuIn(data), angle, CuOut(results))

    return results
end

end



# Testing
using Abstractions
using CUDA

# image = ones(Float32, (10,10))
# results1 = scan(CuArray(image), COLS, SUM)
# results2 = map(results1, SQUARE)
# results3 = reduce(results2, COLS, SUM)
# println("image:")
# println(image)
# # println("results 1 (scan):")
# # println(results1)
# # println("results 2 (map):")
# # println(results2)
# println("results 3 (reduce):")
# println(transpose(to_host(results3)))

image = float32([0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0;
                 0 0 0 0 1 1 1 0 0 0 0])
result = rotate(CuArray(image), float32(0.5))
println("image:")
println(image)
println("result:")
println(to_host(result))
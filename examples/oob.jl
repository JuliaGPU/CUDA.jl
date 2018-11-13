# EXCLUDE FROM TESTING
# this example might fail (CUDA error, or runtime trap if bounds-checking if enabled)

# Running this example under cuda-memset properly gives line number info,
# demonstrating how we support existing CUDA tools.

# TODO: make the actual error trap at run time

using CUDAdrv, CUDAnative, CuArrays

a = CuArray{Float32}(10)

function memset(a, val)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    a[i] = val
    return
end

@cuda threads=11 memset(a, 0f0)
synchronize()

using CUDAdrv, CUDAnative

@noinline function child(i)
    if i < 10
        return i*i
    else
        return (i-1)*(i+1)
    end
end

function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + child(b[i])

    return nothing
end

dev = CuDevice(0)
ctx = CuContext(dev)

dims = (3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

len = prod(dims)
@cuda (1,len) kernel_vadd(d_a, d_b, d_c)
c = Array(d_c)

CUDAnative.code_sass(kernel_vadd, Tuple{CuDeviceArray{Float32,2}, CuDeviceArray{Float32,2}, CuDeviceArray{Float32,2}})
CUDAnative.code_ptx(child, Tuple{Float32})
CUDAnative.code_sass(child, Tuple{Float32})

destroy(ctx)

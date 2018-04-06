using CUDAdrv, CUDAnative

using Test

function vadd(gpu, a, b, c)
    i = threadIdx().x + blockDim().x * ((blockIdx().x-1) + (gpu-1) * gridDim().x)
    c[i] = a[i] + b[i]
end

gpus = Int(length(devices()))

dims = (gpus,3,4)
a = round.(rand(Float32, dims) * 100)
b = round.(rand(Float32, dims) * 100)

# FIXME: CuArray doesn't tie in with unified memory yet
buf_a = Mem.alloc(sizeof(a), true)
Mem.upload!(buf_a, a)
d_a = CuArray{Float32,3}(dims, buf_a)
buf_b = Mem.alloc(sizeof(a), true)
Mem.upload!(buf_b, b)
d_b = CuArray{Float32,3}(dims, buf_b)
buf_c = Mem.alloc(sizeof(a), true)
d_c = CuArray{Float32,3}(dims, buf_c)

len = prod(dims)
blocks = gpus
threads = len ÷ blocks

for (gpu,dev) in enumerate(devices())
    @info "Allocating slice $gpu on device $(name(dev))"
    device!(dev)
    @cuda blocks=blocks÷gpus threads=threads vadd(gpu, d_a, d_b, d_c)
end

@info "Synchronizing devices"
for dev in devices()
    # NOTE: normally you'd use events and wait for them
    device!(dev)
    synchronize()
end

c = Array(d_c)
@test a+b ≈ c

using CUDA
using CUDA: i32

using Test

"Dummy kernel doing 100 FMAs."
function kernel_100fma(a, b, c, out)
    i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
    @inbounds if i <= length(out)
        a_val = a[i]
        b_val = b[i]
        c_val = c[i]

        for j in 1:33
            a_val = CUDA.fma(a_val, b_val, c_val)
            b_val = CUDA.fma(a_val, b_val, c_val)
            c_val = CUDA.fma(a_val, b_val, c_val)
        end

        out[i] = CUDA.fma(a_val, b_val, c_val)
    end

    return
end

function peakflops(n::Integer=5000, dev::CuDevice=CuDevice(0))
    device!(dev) do
        dims = (n, n)
        a = round.(rand(Float32, dims) * 100)
        b = round.(rand(Float32, dims) * 100)
        c = round.(rand(Float32, dims) * 100)
        out = similar(a)

        d_a = CuArray(a)
        d_b = CuArray(b)
        d_c = CuArray(c)
        d_out = CuArray(out)

        len = prod(dims)

        kernel = @cuda launch=false kernel_100fma(d_a, d_b, d_c, d_out)
        config = launch_configuration(kernel.fun)
        threads = min(len, config.threads)
        blocks = cld(len, threads)

        # warm-up
        kernel(d_a, d_b, d_c, d_out)
        synchronize()

        secs = CUDA.@elapsed begin
            kernel(d_a, d_b, d_c, d_out; threads=threads, blocks=blocks)
        end
        flopcount = 200*len
        flops = flopcount / secs

        return flops
    end
end

println(peakflops())

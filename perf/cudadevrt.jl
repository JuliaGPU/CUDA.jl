module cudadevrt

using CUDA, BenchmarkTools, Random

const threads = 256
#simple add matrix and vector kernel
function kernel_add_mat_vec(m, x1, x2, y)
    # one block per column
    offset = (blockIdx().x-1) * m
    @inbounds xtmp = x2[blockIdx().x]
    for i = threadIdx().x : blockDim().x : m
        @inbounds y[offset + i] = x1[offset + i] + xtmp
    end
    return
end

function add!(y, x1, x2)
    m, n = size(x1)
    @cuda blocks = n, 1 threads = threads kernel_add_mat_vec(m, x1, x2, y)
end

function main()
    Random.seed!(1)
    m, n = 3072, 1536    # 256 multiplier
    x1 = cu(randn(Float32, (m, n)) .+ Float32(0.5))
    x2 = cu(randn(Float32, (1, n)) .+ Float32(0.5))
    y1 = similar(x1)

    results = @benchmark CUDA.@sync add!($y1, $x1, $x2)

    # BenchmarkTools captures inputs, JuliaCI/BenchmarkTools.jl#127, so forcibly free them
    CUDA.unsafe_free!(x1)
    CUDA.unsafe_free!(x2)
    CUDA.unsafe_free!(y1)

    return results
end

end

cudadevrt.main()


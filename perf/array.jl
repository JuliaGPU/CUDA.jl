group = addgroup!(SUITE, "array")

const m = 512
const n = 1000

# generate some arrays
cpu_mat = rand(rng, Float32, m, n)
gpu_mat = CuArray{Float32}(undef, size(cpu_mat))
gpu_vec = reshape(gpu_mat, length(gpu_mat))
gpu_mat_ints = CuArray(rand(rng, Int, m, n))
gpu_vec_ints = reshape(gpu_mat_ints, length(gpu_mat_ints))
gpu_mat_bools = CuArray(rand(rng, Bool, m, n))
gpu_vec_bools = reshape(gpu_mat_bools, length(gpu_mat_bools))

group["construct"] = @benchmarkable CuArray{Int}(undef, 1)

group["copy"] = @async_benchmarkable copy($gpu_mat)

gpu_mat2 = copy(gpu_mat)
let group = addgroup!(group, "copyto!")
    group["cpu_to_gpu"] = @async_benchmarkable copyto!($gpu_mat, $cpu_mat)
    group["gpu_to_cpu"] = @async_benchmarkable copyto!($cpu_mat, $gpu_mat)
    group["gpu_to_gpu"] = @async_benchmarkable copyto!($gpu_mat2, $gpu_mat)
end

let group = addgroup!(group, "iteration")
    group["scalar"] = @benchmarkable CUDA.@allowscalar [$gpu_vec[i] for i in 1:10]

    group["logical"] = @benchmarkable $gpu_vec[$gpu_vec_bools]

    group["findall"] = @benchmarkable findall($gpu_vec_bools)
    group["findall"] = @benchmarkable findall(isodd, $gpu_vec_ints)

    group["findfirst"] = @benchmarkable findfirst($gpu_vec_bools)
    group["findfirst"] = @benchmarkable findfirst(isodd, $gpu_vec_ints)

    let group = addgroup!(group, "findmin") # findmax
        group["1d"] = @async_benchmarkable findmin($gpu_vec)
        group["2d"] = @async_benchmarkable findmin($gpu_mat; dims=1)
    end
end

let group = addgroup!(group, "reverse")
    group["1d"] = @async_benchmarkable reverse($gpu_vec)
    group["2d"] = @async_benchmarkable reverse($gpu_mat; dims=1)
    group["1d_inplace"] = @async_benchmarkable reverse!($gpu_vec)
    group["2d_inplace"] = @async_benchmarkable reverse!($gpu_mat; dims=1)
end

group["broadcast"] = @async_benchmarkable $gpu_mat .= 0f0

# no need to test inplace version, which performs the same operation (but with an alloc)
let group = addgroup!(group, "accumulate")
    group["1d"] = @async_benchmarkable accumulate(+, $gpu_vec)
    group["2d"] = @async_benchmarkable accumulate(+, $gpu_mat; dims=1)
end

let group = addgroup!(group, "reductions")
    let group = addgroup!(group, "reduce")
        group["1d"] = @async_benchmarkable reduce(+, $gpu_vec)
        group["2d"] = @async_benchmarkable reduce(+, $gpu_mat; dims=1)
    end

    let group = addgroup!(group, "mapreduce")
        group["1d"] = @async_benchmarkable mapreduce(x->x+1, +, $gpu_vec)
        group["2d"] = @async_benchmarkable mapreduce(x->x+1, +, $gpu_mat; dims=1)
    end

    # used by sum, prod, minimum, maximum, all, any, count
end

let group = addgroup!(group, "random")
    let group = addgroup!(group, "rand")
        group["Float32"] = @async_benchmarkable CUDA.rand(Float32, m*n)
        group["Int64"] = @async_benchmarkable CUDA.rand(Int64, m*n)
    end

    let group = addgroup!(group, "rand!")
        group["Float32"] = @async_benchmarkable CUDA.rand!($gpu_vec)
        group["Int64"] = @async_benchmarkable CUDA.rand!($gpu_vec_ints)
    end

    let group = addgroup!(group, "randn")
        group["Float32"] = @async_benchmarkable CUDA.randn(Float32, m*n)
        #group["Int64"] = @async_benchmarkable CUDA.randn(Int64, m*n)
    end

    let group = addgroup!(group, "randn!")
        group["Float32"] = @async_benchmarkable CUDA.randn!($gpu_vec)
        #group["Int64"] = @async_benchmarkable CUDA.randn!($gpu_vec_ints)
    end
end

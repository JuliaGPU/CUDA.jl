group = addgroup!(SUITE, "array")

# generate some arrays
cpu_mat = rand(Float32, 512, 1000)
gpu_mat = CuArray{Float32}(undef, size(cpu_mat))
gpu_vec = reshape(gpu_mat, length(gpu_mat))
gpu_mat_ints = CuArray(rand(Int, 512, 1000))
gpu_vec_ints = reshape(gpu_mat_ints, length(gpu_mat_ints))
gpu_mat_bools = CuArray(rand(Bool, 512, 1000))
gpu_vec_bools = reshape(gpu_mat_bools, length(gpu_mat_bools))

group["construct"] = @benchmarkable CuArray{Int}(undef, 1)

group["copy"] = @benchmarkable CUDA.@sync copy($gpu_mat)

let group = addgroup!(group, "copyto!")
    gpu_mat2 = copy(gpu_mat)
    group["cpu_to_gpu"] = @benchmarkable CUDA.@sync copyto!($gpu_mat, $cpu_mat)
    group["gpu_to_cpu"] = @benchmarkable CUDA.@sync copyto!($cpu_mat, $gpu_mat)
    group["gpu_to_gpu"] = @benchmarkable CUDA.@sync copyto!($gpu_mat2, $gpu_mat)
end

let group = addgroup!(group, "iteration")
    group["scalar"] = @benchmarkable CUDA.@allowscalar [$gpu_vec[i] for i in 1:10]

    group["logical"] = @benchmarkable $gpu_vec[$gpu_vec_bools]

    group["findall"] = @benchmarkable findall($gpu_vec_bools)
    group["findall"] = @benchmarkable findall(isodd, $gpu_vec_ints)

    group["findfirst"] = @benchmarkable findfirst($gpu_vec_bools)
    group["findfirst"] = @benchmarkable findfirst(isodd, $gpu_vec_ints)

    let group = addgroup!(group, "findmin") # findmax
        group["1d"] = @benchmarkable CUDA.@sync findmin($gpu_vec)
        group["2d"] = @benchmarkable CUDA.@sync findmin($gpu_mat; dims=1)
    end
end

let group = addgroup!(group, "reverse")
    group["1d"] = @benchmarkable CUDA.@sync reverse($gpu_vec)
    group["2d"] = @benchmarkable CUDA.@sync reverse($gpu_mat; dims=1)
    group["1d_inplace"] = @benchmarkable CUDA.@sync reverse!($gpu_vec)
    group["2d_inplace"] = @benchmarkable CUDA.@sync reverse!($gpu_mat; dims=1)
end

group["broadcast"] = @benchmarkable CUDA.@sync $gpu_mat .= 0f0

# no need to test inplace version, which performs the same operation (but with an alloc)
let group = addgroup!(group, "accumulate")
    group["1d"] = @benchmarkable CUDA.@sync accumulate(+, $gpu_vec)
    group["2d"] = @benchmarkable CUDA.@sync accumulate(+, $gpu_mat; dims=1)
end

let group = addgroup!(group, "reductions")
    let group = addgroup!(group, "reduce")
        group["1d"] = @benchmarkable CUDA.@sync reduce(+, $gpu_vec)
        group["2d"] = @benchmarkable CUDA.@sync reduce(+, $gpu_mat; dims=1)
    end

    let group = addgroup!(group, "mapreduce")
        group["1d"] = @benchmarkable CUDA.@sync mapreduce(x->x+1, +, $gpu_vec)
        group["2d"] = @benchmarkable CUDA.@sync mapreduce(x->x+1, +, $gpu_mat; dims=1)
    end

    # used by sum, prod, minimum, maximum, all, any, count
end

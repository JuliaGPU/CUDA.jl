group = addgroup!(SUITE, "array")

group["construct"] = @benchmarkable CuArray{Int}(undef, 1)

cpu_mat = rand(Float32, 512, 1000)
gpu_mat = CuArray{Float32}(undef, size(cpu_mat))
group["copy"] = @benchmarkable CUDA.@sync copy($gpu_mat)

gpu_mat2 = copy(gpu_mat)
let group = addgroup!(SUITE, "copyto!")
    group["cpu_to_gpu"] = @benchmarkable CUDA.@sync copyto!($gpu_mat, $cpu_mat)
    group["gpu_to_cpu"] = @benchmarkable CUDA.@sync copyto!($cpu_mat, $gpu_mat)
    group["gpu_to_gpu"] = @benchmarkable CUDA.@sync copyto!($gpu_mat2, $gpu_mat)
end

group["fill!"] = @benchmarkable CUDA.@sync fill!($gpu_mat, 0f0)

gpu_vec = reshape(gpu_mat, length(gpu_mat))
let group = addgroup!(SUITE, "reverse")
    group["1d"] = @benchmarkable CUDA.@sync reverse($gpu_vec)
    group["2d"] = @benchmarkable CUDA.@sync reverse($gpu_mat2; dims=1)
    group["1d_inplace"] = @benchmarkable CUDA.@sync reverse!($gpu_vec)
    group["2d_inplace"] = @benchmarkable CUDA.@sync reverse!($gpu_mat2; dims=1)
end

group["broadcast"] = @benchmarkable CUDA.@sync $gpu_mat .= 0f0

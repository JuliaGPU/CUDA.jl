# FFT plan creation for common types/dimensions
for T in (ComplexF32, ComplexF64)
    precompile(Tuple{typeof(plan_fft!), CUDACore.CuArray{T, 1}, Tuple{Int}})
    precompile(Tuple{typeof(plan_fft!), CUDACore.CuArray{T, 2}, Tuple{Int, Int}})
end
for T in (Float32, Float64)
    precompile(Tuple{typeof(plan_rfft), CUDACore.CuArray{T, 1}, Tuple{Int}})
    precompile(Tuple{typeof(plan_rfft), CUDACore.CuArray{T, 2}, Tuple{Int, Int}})
end

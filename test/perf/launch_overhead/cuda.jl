#!/usr/bin/env julia

# CUDAdrv.jl version

using CUDAdrv

const len = 1000
const ITERATIONS = 5000

# TODO: api-trace shows some attribute fetches, where do they come from?

function main()    
    mod = CuModuleFile("cuda.ptx")
    fun = CuFunction(mod, "kernel_dummy")

    cpu_time = Vector{Float64}(ITERATIONS)
    gpu_time = Vector{Float64}(ITERATIONS)

    gpu_arr = CuArray{Float32}(len)
    for i in 1:ITERATIONS
        i == ITERATIONS-4 && CUDAdrv.Profile.start()

        gpu_tic, gpu_toc = CuEvent(), CuEvent()

        cpu_tic = time_ns()
        record(gpu_tic)
        cudacall(fun, len, 1, (Ptr{Float32},), pointer(gpu_arr))
        record(gpu_toc)
        synchronize(gpu_toc)
        cpu_toc = time_ns()

        cpu_time[i] = (cpu_toc-cpu_tic)/1000
        gpu_time[i] = CUDAdrv.elapsed(gpu_tic, gpu_toc)*1000000
    end
    CUDAdrv.Profile.stop()

    @printf("CPU time: %.2fus\n", median(cpu_time))
    @printf("GPU time: %.2fus\n", median(gpu_time))
end

main()

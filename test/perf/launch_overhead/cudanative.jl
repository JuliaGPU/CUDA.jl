#!/usr/bin/env julia

# CUDAnative.jl version

using CUDAdrv, CUDAnative
using InteractiveUtils

function kernel_dummy(ptr)
    Base.pointerset(ptr, 0f0, Int(blockIdx().x), 8)
    return
end

const len = 1000
const ITERATIONS = 5000

function benchmark(gpu_buf)
    @cuda threads=len kernel_dummy(Base.unsafe_convert(Ptr{Float32}, gpu_buf))
    return
end

function main()    
    cpu_time = Vector{Float64}(ITERATIONS)
    gpu_time = Vector{Float64}(ITERATIONS)

    gpu_buf = Mem.alloc(len*sizeof(Float32))
    @code_warntype benchmark(gpu_buf)
    for i in 1:ITERATIONS
        i == ITERATIONS-4 && CUDAdrv.Profile.start()

        gpu_tic, gpu_toc = CuEvent(), CuEvent()

        cpu_tic = time_ns()
        record(gpu_tic)
        benchmark(gpu_buf)
        record(gpu_toc)
        synchronize(gpu_toc)
        cpu_toc = time_ns()

        cpu_time[i] = (cpu_toc-cpu_tic)/1000
        gpu_time[i] = CUDAdrv.elapsed(gpu_tic, gpu_toc)*1000000
    end
    CUDAdrv.Profile.stop()
    Mem.free(gpu_buf)

    @printf("CPU time: %.2fus\n", median(cpu_time))
    @printf("GPU time: %.2fus\n", median(gpu_time))
end

main()

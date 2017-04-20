#!/usr/bin/env julia

# CUDAnative.jl version, using the fancy profiler instead of manual events

using CUDAdrv, CUDAnative

function kernel_dummy(ptr)
    Base.pointerset(ptr, Float32(0), Int(blockIdx().x), 8)
    return nothing
end

const len = 1000

const ITERATIONS = 5000

function main()    
    dev = CuDevice(0)
    ctx = CuContext(dev)

    cpu_time = Vector{Float64}(ITERATIONS)

    gpu_arr = CuArray{Float32}(len)
    for i in 1:ITERATIONS
        i == ITERATIONS-4 && CUDAdrv.start_profiler()

        cpu_tic = time_ns()
        CUDAnative.@profile begin
            @cuda (len,1) kernel_dummy(pointer(gpu_arr))
        end
        cpu_toc = time_ns()

        cpu_time[i] = (cpu_toc-cpu_tic)/1000
    end
    CUDAdrv.stop_profiler()

    @printf("CPU time: %.2fus\n", median(cpu_time))
    CUDAnative.Profile.print()

    destroy(ctx)
end

main()

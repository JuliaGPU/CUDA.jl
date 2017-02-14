#!/usr/bin/env julia

using CUDAdrv, CUDAnative

function kernel_dummy(ptr)
    Base.pointerset(ptr, Float32(0), Int(blockIdx().x), 8)
    return nothing
end

const len = 100000

const ITERATIONS = 5000

function main()    
    dev = CuDevice(0)
    ctx = CuContext(dev)

    cpu_time = Vector{Float64}(ITERATIONS)
    gpu_time = Vector{Float64}(ITERATIONS)

    gpu_arr = CuArray{Float32}(len)
    for i in 1:ITERATIONS
        i == ITERATIONS-4 && CUDAdrv.start_profiler()
        cpu_time[i] = Base.@elapsed begin
            gpu_time[i] = CUDAdrv.@elapsed begin
                @cuda (len,1) kernel_dummy(pointer(gpu_arr))
            end
        end
    end
    CUDAdrv.stop_profiler()

    overhead = cpu_time .- gpu_time
    @printf("Overhead: %.2fus on %.2fus (%.2f%%)\n",
            median(overhead)*1000000, median(gpu_time)*1000000,
            100*median(overhead)/median(gpu_time))

    destroy(ctx)
end

# @code_warntype main()
# code_llvm(STDOUT, main, (), false)
main()

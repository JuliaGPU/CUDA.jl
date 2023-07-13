# # Performance Tips

# Profile your code! See the [Profiling](../development/profiling.md) page for more details.

# ### Resources

# NVidia's technical blog has a lot of good tips: [Pro-Tips](https://developer.nvidia.com/blog/tag/pro-tip/), [Optimization](https://developer.nvidia.com/blog/tag/optimization/).

# The following notebooks also have some good tips: [JuliaCon 2021 GPU Workshop](https://github.com/maleadt/juliacon21-gpu_workshop/blob/main/deep_dive/CUDA.ipynb), [Advanced Julia GPU Training](https://github.com/JuliaComputing/Training/tree/master/AdvancedGPU)

# Also see the [perf](https://github.com/JuliaGPU/CUDA.jl/tree/master/perf) folder for some optimised code examples.

# ### Julia Specific Tips

# #### Inlinining

# Inlining can reduce register usage and thus speed up kernels. To force inlining of all functions use `@cuda always_inline=true`.

# #### Minimise Runtime Exceptions

# Many common operations can throw errors at runtime in Julia, they often do this by branching and calling a function in that branch both of which are slow on GPUs. Using `@inbounds` when indexing into arrays will eliminate exceptions due to bounds checking. You can also use `assume` from the package LLVM.jl to get rid of exceptions

# ```julia
# using LLVM, LLVM.Interop

# function test(x, y)
#     assume(x > 0)
#     div(y, x)
# end
# ```

# The `assume(x > 0)` tells the compiler that there cannot be a divide by 0 error.

# For more information and examples check out [Kernel analysis and optimization](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb).

# #### 32-bit Integers

# Use 32-bit integers where possible. A common source of register pressure is the use of 64-bit integers when only 32-bits are required. For example, the hardware's indices are 32-bit integers, but Julia's literals are Int64's which results in expressions like blockIdx().x-1 to be promoted to 64-bit integers. We can instead replace the `1` to a 32-bit integer with `Int32(1)` or more succintly `1i32` if you run `using CUDA: i32`

# To see how much of a difference this actually makes lets use a kernel introduced in the introduction for inplace addition.

using CUDA, BenchmarkTools

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

# Now let's benchmark it:

x_d = CUDA.fill(1.0f0, 2^28)
y_d = CUDA.fill(2.0f0, 2^28)

function bench_gpu3!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

# ```julia
# @btime bench_gpu3!($y_d, $x_d)
# ```

# ```
#   76.202 ms (58 allocations: 3.73 KiB)
# ```

# Our kernel using 32-bit integers is below

function gpu_add4!(y, x)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add4!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

# ```julia
# @btime bench_gpu4!($y_d, $x_d)
# ```

# ```
#   76.149 ms (57 allocations: 3.70 KiB)
# ```

# As we can see from the benchmarks we only get a small performance benefit but there can be larger differences for more complex kernels.

# #### Avoiding StepRange

# In the previous kernel in the for loop we iterated over `index:stride:length(y)`, this is a `StepRange`. Unfortunately, constructing a `StepRange` is slow as they can throw errors and there is some unnecessary calculation when we just want to loop over them. Instead it is faster to use a while loop like so:

function gpu_add5!(y, x)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    i = index
    while i <= length(y)
        @inbounds y[i] += x[i]
        i += stride
    end
    return
end

# The benchmark[^1]:

function bench_gpu5!(y, x)
    kernel = @cuda launch=false gpu_add5!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

# ```julia
# @btime bench_gpu5!($y_d, $x_d)
# ```

# ```
#   75.732 ms (58 allocations: 3.73 KiB)
# ```

# This benchmark shows there is a only a small performance benefit for this kernel however we can see a big difference in the amount of registers used:

# ```julia
# CUDA.registers(@cuda gpu_add4!(y_d, x_d))
# ```

# ```
#   28
# ```

# ```julia
# CUDA.registers(@cuda gpu_add5!(y_d, x_d))
# ```

# ```
#   12
# ```

# [^1]: Conducted on Julia Version 1.9.2, the benefit of this technique should be reduced on version 1.10 or by using `always_inline=true` on the `@cuda` macro, e.g. `@cuda always_inline=true launch=false gpu_add4!(y, x)`.
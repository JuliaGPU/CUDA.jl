# # Performance Tips

# ## General Tips

# Always start by profiling your code (see the [Profiling](../development/profiling.md) page for more details). You first want to analyze your application as a whole, using `CUDA.@profile` or NSight Systems, identifying hotspots and bottlenecks. Focusing on these you will want to:

# * Minimize data transfer between the CPU and GPU, you can do this by getting rid of unnecessary memory copies and batching many small transfers into larger ones;
# * Identify problematic kernel invocations: you may be launching thousands of kernels which could be fused into a single call;
# * Find stalls, where the CPU isn't submitting work fast enough to keep the GPU busy.

# If that isn't sufficient, and you identified a kernel that executes slowly, you can try using NSight Compute to analyze that kernel in detail. Some things to try in order of importance:
# * Optimize memory accesses, e.g., avoid needless global accesses (buffering in shared memory instead) or coalesce accesses;
# * Launch more threads on each streaming multiprocessor, this can be achieved by lowering register pressure or reducing shared memory usage, the tips below outline the various ways in which register pressure can be reduced;
# * Use 32 bit types like `Float32` and `Int32` instead of 64 bit types like `Float64` and `Int`/`Int64`;
# * Avoid the use of control flow which cause threads in the same warp to diverge, i.e., make sure `while` or `for` loops behave identically across the entire warp, and replace `if`s that diverge within a warp with `ifelse`s;
# * Increase the arithmetic intensity in order for the GPU to be able to hide the latency of memory accesses.

# ### Inlining

# Inlining can reduce register usage and thus speed up kernels. To force inlining of all functions use `@cuda always_inline=true`.

# ### Limiting the Maximum Number of Registers Per Thread

# The number of threads that can be launched is partly determined by the number of registers a kernel uses. This is due to registers being shared between all threads on a multiprocessor.
# Setting the maximum number of registers per thread will force less registers to be used which can increase thread count at the expense of having to spill registers into local memory, this may improve performance. To set the max registers to 32 use `@cuda maxregs=32`.

# ### FastMath

# Use `@fastmath` to use faster versions of common mathematical functions and use `@cuda fastmath=true` for even faster square roots.

# ## Resources

# For further information you can check out these resources.

# NVidia's technical blog has a lot of good tips: [Pro-Tips](https://developer.nvidia.com/blog/tag/pro-tip/), [Optimization](https://developer.nvidia.com/blog/tag/optimization/).

# The [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) is relevant for Julia.

# The following notebooks also have some good tips: [JuliaCon 2021 GPU Workshop](https://github.com/maleadt/juliacon21-gpu_workshop/blob/main/deep_dive/CUDA.ipynb), [Advanced Julia GPU Training](https://github.com/JuliaComputing/Training/tree/master/AdvancedGPU).

# Also see the [perf](https://github.com/JuliaGPU/CUDA.jl/tree/master/perf) folder for some optimised code examples.

# ## Julia Specific Tips

# ### Minimise Runtime Exceptions

# Many common operations can throw errors at runtime in Julia, they often do this by branching and calling a function in that branch both of which are slow on GPUs. Using `@inbounds` when indexing into arrays will eliminate exceptions due to bounds checking. Note that running code with `--check-bounds=yes` (the default for `Pkg.test`) will always emit bounds checking. You can also use `assume` from the package LLVM.jl to get rid of exceptions, e.g.

# ```julia
# using LLVM.Interop

# function test(x, y)
#     assume(x > 0)
#     div(y, x)
# end
# ```

# The `assume(x > 0)` tells the compiler that there cannot be a divide by 0 error.

# For more information and examples check out [Kernel analysis and optimization](https://github.com/JuliaComputing/Training/blob/master/AdvancedGPU/2-2-kernel_analysis_optimization.ipynb).

# ### 32-bit Integers

# Use 32-bit integers where possible. A common source of register pressure is the use of 64-bit integers when only 32-bits are required. For example, the hardware's indices are 32-bit integers, but Julia's literals are Int64's which results in expressions like `blockIdx().x-1` to be promoted to 64-bit integers. To use 32-bit integers we can instead replace the `1` with `Int32(1)` or more succintly `1i32` if you run `using CUDA: i32`.

# To see how much of a difference this makes let's use a kernel introduced in the introductory tutorial for inplace addition.

using CUDA, BenchmarkTools

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

# Now let's see how many registers are used:

# ```julia
# x_d = CUDA.fill(1.0f0, 2^28)
# y_d = CUDA.fill(2.0f0, 2^28)
#
# CUDA.registers(@cuda gpu_add3!(y_d, x_d))
# ```

# ```
#   29
# ```

# Our kernel using 32-bit integers is below:

function gpu_add4!(y, x)
    index = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

# ```julia
# CUDA.registers(@cuda gpu_add4!(y_d, x_d))
# ```

# ```
#   28
# ```

# So we use one less register by switching to 32 bit integers, for kernels using even more 64 bit integers we would expect to see larger falls in register count.

# ### Avoiding `StepRange`

# In the previous kernel in the for loop we iterated over `index:stride:length(y)`, this is a `StepRange`. Unfortunately, constructing a `StepRange` is slow as they can throw errors and they contain unnecessary computation when we just want to loop over them. Instead it is faster to use a while loop like so:

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

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add4!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync kernel(y, x; threads, blocks)
end

function bench_gpu5!(y, x)
    kernel = @cuda launch=false gpu_add5!(y, x)
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

# ```julia
# @btime bench_gpu5!($y_d, $x_d)
# ```

# ```
#   75.732 ms (58 allocations: 3.73 KiB)
# ```

# This benchmark shows there is a only a small performance benefit for this kernel however we can see a big difference in the amount of registers used, recalling that 28 registers were used when using a `StepRange`:

# ```julia
# CUDA.registers(@cuda gpu_add5!(y_d, x_d))
# ```

# ```
#   12
# ```

# [^1]: Conducted on Julia Version 1.9.2, the benefit of this technique should be reduced on version 1.10 or by using `always_inline=true` on the `@cuda` macro, e.g. `@cuda always_inline=true launch=false gpu_add4!(y, x)`.

# # Introduction
#
# *A gentle introduction to parallelization and GPU programming in Julia*
#
# [Julia](https://julialang.org/) has first-class support for GPU programming: you can use
# high-level abstractions or obtain fine-grained control, all without ever leaving your
# favorite programming language. The purpose of this tutorial is to help Julia users take
# their first step into GPU computing. In this tutorial, you'll compare CPU and GPU
# implementations of a simple calculation, and learn about a few of the factors that
# influence the performance you obtain.
#
# This tutorial is inspired partly by a blog post by Mark Harris, [An Even Easier
# Introduction to CUDA](https://devblogs.nvidia.com/even-easier-introduction-cuda/), which
# introduced CUDA using the C++ programming language. You do not need to read that
# tutorial, as this one starts from the beginning.



# ## A simple example on the CPU

# We'll consider the following demo, a simple calculation on the CPU.

N = 2^20
x = fill(1.0f0, N)  # a vector filled with 1.0 (Float32)
y = fill(2.0f0, N)  # a vector filled with 2.0

y .+= x             # increment each element of y with the corresponding element of x

# check that we got the right answer
using Test
@test all(y .== 3.0f0)

# From the `Test Passed` line we know everything is in order. We used `Float32` numbers in
# preparation for the switch to GPU computations: GPUs are faster (sometimes, much faster)
# when working with `Float32` than with `Float64`.

# A distinguishing feature of this calculation is that every element of `y` is being
# updated using the same operation. This suggests that we might be able to parallelize
# this.


# ### Parallelization on the CPU

# First let's do the parallelization on the CPU. We'll create a "kernel function" (the
# computational core of the algorithm) in two implementations, first a sequential version:

function sequential_add!(y, x)
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
sequential_add!(y, x)
@test all(y .== 3.0f0)

# And now a parallel implementation:

function parallel_add!(y, x)
    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y, 2)
parallel_add!(y, x)
@test all(y .== 3.0f0)

# Now if I've started Julia with `JULIA_NUM_THREADS=4` on a machine with at least 4 cores,
# I get the following:

@assert Threads.nthreads() == 4     #src

# ```julia
# using BenchmarkTools
# @btime sequential_add!($y, $x)
# ```

# ```
#   487.303 μs (0 allocations: 0 bytes)
# ```

# versus

# ```julia
# @btime parallel_add!($y, $x)
# ```

# ```
#   259.587 μs (13 allocations: 1.48 KiB)
# ```

# You can see there's a performance benefit to parallelization, though not by a factor of 4
# due to the overhead for starting threads. With larger arrays, the overhead would be
# "diluted" by a larger amount of "real work"; these would demonstrate scaling that is
# closer to linear in the number of cores. Conversely, with small arrays, the parallel
# version might be slower than the serial version.



# ## Your first GPU computation

# ### Installation

# For most of this tutorial you need to have a computer with a compatible GPU and have
# installed [CUDA](https://developer.nvidia.com/cuda-downloads). You should also install
# the following packages using Julia's [package
# manager](https://docs.julialang.org/en/latest/stdlib/Pkg/):

# ```julia
# pkg> add CUDA
# ```

# If this is your first time, it's not a bad idea to test whether your GPU is working by
# testing the CUDA.jl package:

# ```julia
# pkg> add CUDA
# pkg> test CUDA
# ```


# ### Parallelization on the GPU

# We'll first demonstrate GPU computations at a high level using the `CuArray` type,
# without explicitly writing a kernel function:

using CUDA

x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

# Here the `d` means "device," in contrast with "host". Now let's do the increment:

y_d .+= x_d
@test all(Array(y_d) .== 3.0f0)

# The statement `Array(y_d)` moves the data in `y_d` back to the host for testing. If we
# want to benchmark this, let's put it in a function:

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

# ```julia
# @btime add_broadcast!($y_d, $x_d)
# ```

# ```
#   67.047 μs (84 allocations: 2.66 KiB)
# ```

# The most interesting part of this is the call to `CUDA.@sync`. The CPU can assign
# jobs to the GPU and then go do other stuff (such as assigning *more* jobs to the GPU)
# while the GPU completes its tasks. Wrapping the execution in a `CUDA.@sync` block
# will make the CPU block until the queued GPU tasks are done, similar to how `Base.@sync`
# waits for distributed CPU tasks. Without such synchronization, you'd be measuring the
# time takes to launch the computation, not the time to perform the computation. But most
# of the time you don't need to synchronize explicitly: many operations, like copying
# memory from the GPU to the CPU, implicitly synchronize execution.

# For this particular computer and GPU, you can see the GPU computation was significantly
# faster than the single-threaded CPU computation, and that the use of multiple CPU threads makes
# the CPU implementation competitive. Depending on your hardware you may get different
# results.


# ### Writing your first GPU kernel

# Using the high-level GPU array functionality made it easy to perform this computation
# on the GPU. However, we didn't learn about what's going on under the hood, and that's the
# main goal of this tutorial. So let's implement the same functionality with a GPU kernel:

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# Aside from using the `CuArray`s `x_d` and `y_d`, the only GPU-specific part of this is the
# *kernel launch* via `@cuda`. The first time you issue this `@cuda` statement, it will
# compile the kernel (`gpu_add1!`) for execution on the GPU. Once compiled, future
# invocations are fast. You can see what `@cuda` expands to using `?@cuda` from the Julia
# prompt.

# Let's benchmark this:

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

# ```julia
# @btime bench_gpu1!($y_d, $x_d)
# ```

# ```
#   119.783 ms (47 allocations: 1.23 KiB)
# ```

# That's a *lot* slower than the version above based on broadcasting. What happened?


# ### Profiling

# When you don't get the performance you expect, usually your first step should be to
# profile the code and see where it's spending its time. For that, you'll need to be able to
# run NVIDIA's [`nvprof`
# tool](https://devblogs.nvidia.com/cuda-pro-tip-nvprof-your-handy-universal-gpu-profiler/).
# On Unix systems, launch Julia this way:
#
# ```sh
# $ nvprof --profile-from-start off /path/to/julia
# ```
#
# replacing the `/path/to/julia` with the path to your Julia binary. Note that we don't
# immediately start the profiler, but instead call into the CUDA APIs and manually start the
# profiler with `CUDA.@profile` (thus excluding the time to compile our kernel):

bench_gpu1!(y_d, x_d)  # run it once to force compilation
CUDA.@profile bench_gpu1!(y_d, x_d)

# When we quit the Julia REPL, the profiler process will print information about the
# executed kernels and API calls:

# ```
# ==2574== Profiling result:
#             Type  Time(%)      Time     Calls       Avg       Min       Max  Name
#  GPU activities:  100.00%  247.61ms         1  247.61ms  247.61ms  247.61ms  ptxcall_gpu_add1__1
#       API calls:   99.54%  247.83ms         1  247.83ms  247.83ms  247.83ms  cuEventSynchronize
#                     0.46%  1.1343ms         1  1.1343ms  1.1343ms  1.1343ms  cuLaunchKernel
#                     0.00%  4.9490us         1  4.9490us  4.9490us  4.9490us  cuEventRecord
#                     0.00%  4.4190us         1  4.4190us  4.4190us  4.4190us  cuEventCreate
#                     0.00%     960ns         2     480ns     358ns     602ns  cuCtxGetCurrent
# ```

# You can see that 100% of the time was spent in `ptxcall_gpu_add1__1`, the name of the
# kernel that CUDA.jl assigned when compiling `gpu_add1!` for these inputs. (Had you created
# arrays of multiple data types, e.g., `xu_d = CUDA.fill(0x01, N)`, you might have also seen
# `ptxcall_gpu_add1__2` and so on. Like the rest of Julia, you can define a single method
# and it will be specialized at compile time for the particular data types you're using.)

# For further insight, run the profiling with the option `--print-gpu-trace`. You can also
# invoke Julia with as argument the path to a file containing all commands you want to run
# (including a call to `CUDA.@profile`):
#
# ```sh
# $ nvprof --profile-from-start off --print-gpu-trace /path/to/julia /path/to/script.jl
#      Start  Duration   Grid Size   Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream  Name
#   13.3134s  245.04ms     (1 1 1)      (1 1 1)        20        0B        0B  GeForce GTX TIT         1         7  ptxcall_gpu_add1__1 [34]
# ```

# The key thing to note here is the `(1 1 1)` in the "Grid Size" and "Block Size" columns.
# These terms will be explained shortly, but for now, suffice it to say that this is an
# indication that this computation ran sequentially. Of note, sequential processing with
# GPUs is much slower than with CPUs; where GPUs shine is with large-scale parallelism.


# ### Writing a parallel GPU kernel

# To speed up the kernel, we want to parallelize it, which means assigning different tasks
# to different threads.  To facilitate the assignment of work, each CUDA thread gets access
# to variables that indicate its own unique identity, much as
# [`Threads.threadid()`](https://docs.julialang.org/en/latest/manual/parallel-computing/#Multi-Threading-(Experimental)-1)
# does for CPU threads. The CUDA analogs of `threadid` and `nthreads` are called
# `threadIdx` and `blockDim`, respectively; one difference is that these return a
# 3-dimensional structure with fields `x`, `y`, and `z` to simplify cartesian indexing for
# up to 3-dimensional arrays. Consequently we can assign unique work in the following way:

function gpu_add2!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

fill!(y_d, 2)
@cuda threads=256 gpu_add2!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# Note the `threads=256` here, which divides the work among 256 threads numbered in a
# linear pattern. (For a two-dimensional array, we might have used `threads=(16, 16)` and
# then both `x` and `y` would be relevant.)

# Now let's try benchmarking it:

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

# ```julia
# @btime bench_gpu2!($y_d, $x_d)
# ```

# ```
#   1.873 ms (47 allocations: 1.23 KiB)
# ```

# Much better!

# But obviously we still have a ways to go to match the initial broadcasting result. To do
# even better, we need to parallelize more. GPUs have a limited number of threads they can
# run on a single *streaming multiprocessor* (SM), but they also have multiple SMs. To take
# advantage of them all, we need to run a kernel with multiple *blocks*. We'll divide up
# the work like this:
#
# ![block grid](intro1.png)
#
# This diagram was [borrowed from a description of the C/C++
# library](https://devblogs.nvidia.com/even-easier-introduction-cuda/); in Julia, threads
# and blocks begin numbering with 1 instead of 0. In this diagram, the 4096 blocks of 256
# threads (making 1048576 = 2^20 threads) ensures that each thread increments just a single
# entry; however, to ensure that arrays of arbitrary size can be handled, let's still use a
# loop:

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

numblocks = ceil(Int, N/256)

fill!(y_d, 2)
@cuda threads=256 blocks=numblocks gpu_add3!(y_d, x_d)
@test all(Array(y_d) .== 3.0f0)

# The benchmark:

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
end

# ```julia
# @btime bench_gpu3!($y_d, $x_d)
# ```

# ```
#   67.268 μs (52 allocations: 1.31 KiB)
# ```

# Finally, we've achieved the similar performance to what we got with the broadcasted
# version. Let's run `nvprof` again to confirm this launch configuration:
#
# ```
# ==23972== Profiling result:
#    Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*           Device   Context    Stream  Name
# 13.3526s  101.22us           (4096 1 1)       (256 1 1)        32        0B        0B  GeForce GTX TIT         1         7  ptxcall_gpu_add3__1 [34]
# ```


# ### Printing

# When debugging, it's not uncommon to want to print some values. This is achieved with
# `@cuprint`:

function gpu_add2_print!(y, x)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    @cuprintln("thread $index, block $stride")
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

@cuda threads=16 gpu_add2_print!(y_d, x_d)
synchronize()

# Note that the printed output is only generated when synchronizing the entire GPU with
# `synchronize()`. This is similar to `CUDA.@sync`, and is the counterpart of
# `cudaDeviceSynchronize` in CUDA C++.


# ### Error-handling

# The final topic of this intro concerns the handling of errors. Note that the kernels
# above used `@inbounds`, but did not check whether `y` and `x` have the same length. If
# your kernel does not respect these bounds, you will run into nasty errors:

# ```
# ERROR: CUDA error: an illegal memory access was encountered (code #700, ERROR_ILLEGAL_ADDRESS)
# Stacktrace:
#  [1] ...
# ```

# If you remove the `@inbounds` annotation, instead you get
#
# ```
# ERROR: a exception was thrown during kernel execution.
#        Run Julia on debug level 2 for device stack traces.
# ```

# As the error message mentions, a higher level of debug information will result in a more
# detailed report. Let's run the same code with with `-g2`:
#
# ```
# ERROR: a exception was thrown during kernel execution.
# Stacktrace:
#  [1] throw_boundserror at abstractarray.jl:484
#  [2] checkbounds at abstractarray.jl:449
#  [3] setindex! at /home/tbesard/Julia/CUDA/src/device/array.jl:79
#  [4] some_kernel at /tmp/tmpIMYANH:6
# ```

# !!! warning
#
#     On older GPUs (with a compute capability below `sm_70`) these errors are fatal,
#     and effectively kill the CUDA environment. On such GPUs, it's often a good idea to
#     perform your "sanity checks" using code that runs on the CPU and only turn over the
#     computation to the GPU once you've deemed it to be safe.



# ## Summary

# Keep in mind that the high-level functionality of CUDA often means that you don't
# need to worry about writing kernels at such a low level. However, there are many cases
# where computations can be optimized using clever low-level manipulations. Hopefully, you
# now feel comfortable taking the plunge.

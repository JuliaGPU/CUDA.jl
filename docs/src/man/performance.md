# Performance

GPU code written in CUDAnative.jl can be as fast or even outperform CUDA C compiled with
`nvcc` (on the condition that the same hardware features are used). This section will
describe how to do so, and what to be careful about.


## Profiling

When optimizing code, it is important to know what to optimize. Luckily, the CUDA toolkit
ships an excellent profiler, `nvprof`, with `nvvp` as the Eclipse-based UI. The CUDAnative
compiler is fully compatible with these tools, and generates the required line number
information to debug performance issues. To generate line number information, invoke Julia
with the command-line option `-g1` (the default option). Using `-g2` puts the PTX JIT in
debug mode, which significantly lowers performance of GPU code and currently does not
improve debugging.

Traces collected with these tools might be very large and sparse, because they capture the
entire application including e.g. kernel compilation or initial data uploads. To avoid this,
run the above profilers with the option "Start profiling at application start" disabled
(`--profile-from-start off` with `nvprof`), make your application perform a warm-up
iteration, and wrap subsequent iterations with `CUDAdrv.@profile`. This macro instructs any
active profiler to start collecting information, resulting in much more focused traces.

For true source-level profiling akin to `Base.@profile`, look at `nvvp`'s PC Sampling View
(requires compute capability >= 5.2, CUDA >= 7.5). In the future, we might have a
`CUDAnative.@profile` offering similar functionality, using the NVIDIA CUPTI library.


## Optimizing

This section is a WIP. Some things to consider:

* `Float64` is expensive, but literal floats are `Float64`. Use `...f0` or cast.
* Same for integers; although the performance hit is small, it increases register pressure.

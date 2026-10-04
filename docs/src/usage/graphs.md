# [Graphs](@id UsageGraphs)

Every operation that CUDA.jl submits to the GPU, like launching a kernel, has a CPU cost.
For workloads that consist of many short operations, that cost can exceed the time the GPU
needs to execute them, leaving the GPU idle. CUDA graphs solve this by recording a sequence
of operations once, and then launching the entire sequence at once, at a fraction of the
CPU cost.

For example, a small multi-layer perceptron evaluated at batch size 16 (four layers of
`mul!` followed by a broadcast) takes about 50 µs per evaluation when executed normally,
but only 16 µs when it is launched as a graph, with the CPU being busy for just 8 µs of
that time. Integrating a small ODE for 10 steps with RK4 (80 broadcast kernels) goes from
260 µs to 70 µs, with only 3.5 µs of CPU time.


## Capturing and launching a graph

The easiest way to create a graph is to capture the operations that some code performs
using [`capture`](@ref). Capturing doesn't execute those operations, so instantiate the
graph to create an executable graph, which can then be launched many times:

```julia
using CUDA

function step!(u, v)
    u .= u .* 0.99f0 .+ v
    v .= v .* 0.5f0
end

u = CUDA.rand(Float32, 1024)
v = CUDA.rand(Float32, 1024)

step!(u, v)         # run once, compiling kernels and initializing libraries

graph = capture() do
    for i in 1:10
        step!(u, v)
    end
end
exec = instantiate(graph)

for i in 1:100
    exec()          # or `launch(exec)`
end
```

Instantiating a graph is relatively expensive, so you should reuse the executable graph.
It's a good idea to execute the code once before capturing it: kernel compilation is
supported during capture, but some libraries need to be initialized first, which isn't
possible while capturing.


## What a graph captures

A graph captures the GPU operations, along with all their arguments. Launching the graph
performs exactly the same operations again: it does not execute the Julia code that was
captured. This has some important consequences:

- **Scalars are fixed.** If a captured kernel took a scalar argument, launching the graph
  uses the value that was captured. To vary a value between launches, store it in GPU
  memory, e.g., in a single-element array that you `fill!` before launching the graph,
  or in pinned host memory (`CuArray{T,N,CUDA.HostMemory}`) that you write from the CPU.
- **Arrays are fixed, but not their contents.** Launching the graph operates on the same
  arrays as the captured operations did. You can change the contents of those arrays
  between launches, e.g., by copying new inputs into them, and read results from them
  after the graph has executed.
- **Host code doesn't execute.** Control flow, printing, and any other CPU work in the
  captured code only happens while capturing.
- **Data copied from CPU memory is copied when capturing.** For example, `copyto!(a, h)`
  with `h` an `Array` copies the contents `h` had while capturing every time the graph is
  launched, and the scalar arguments that are passed to library functions keep their
  captured values. Copying data to CPU memory isn't supported at all. To transfer data
  between the CPU and the GPU when launching a graph, use arrays that are backed by pinned
  host memory, e.g., `CuArray{Float32,1,CUDA.HostMemory}`, which the CPU can access
  directly (after synchronizing).

If you want to execute the same operations on different arrays, either capture a graph for
each set of arrays (e.g., two graphs to alternate between double buffers), or update the
executable graph as explained below.


## Memory

Graphs integrate with CUDA.jl's memory management:

- A graph, and every executable graph instantiated from it, keeps all memory that its
  operations use alive. Arrays that are freed, e.g., by the garbage collector or with
  `unsafe_free!`, are only actually released when the graph is destroyed. Launching a
  graph is like launching a kernel: it takes ownership of the memory it uses, so that
  using that memory from other tasks synchronizes correctly.
- Arrays that are allocated during capture are allocated once, and every launch of the
  graph overwrites their contents. That makes it possible to capture code that allocates,
  like `y .= A * x .+ b` (where `A * x` allocates a temporary array), or code that returns
  a new array: after launching the graph, that array contains the result of the latest
  launch. Graphs do not reuse the memory of temporary arrays though, so for large problems
  it is more efficient to capture code that operates in place.
- The garbage collector keeps working during capture, but memory is only released after
  the capture has ended.


## Updating graphs

When the operations to perform change, e.g., because they need to use different arrays,
you can capture a new graph and use it to [`update`](@ref) an existing executable graph.
That's much cheaper than instantiating a new one, but it only works if the new graph has
the same structure:

```julia
graph = capture(() -> step!(u2, v2))
update(exec, graph)
```

The [`@captured`](@ref) macro automates this: it captures the code it is applied to every
time it executes, and launches it using an executable graph that's updated as needed. As
that still executes the captured code every time, it doesn't save as much CPU time as
launching a graph does.

For finer-grained control, [`update!`](@ref) updates a single operation, using a node
returned by [`capture!`](@ref) (see below).


## Constructing graphs

Captured operations execute one after another. To create graphs whose operations can
execute concurrently, use [`capture!`](@ref) to add operations to a graph, and specify
which operations they depend on:

```julia
graph = CuGraph()
a = capture!(graph) do
    x .= sin.(x)
end
b = capture!(graph) do
    y .= cos.(y)
end
capture!(graph; after=[a; b]) do
    z .= x .+ y
end
```

Here, the first two operations can execute at the same time, while the third waits for
both of them to finish. A graph can be visualized by displaying it as `text/vnd.graphviz`,
e.g., using `show(stdout, MIME"text/vnd.graphviz"(), graph)`.


## Limitations

Not everything can be captured. The following results in a [`CaptureError`](@ref) or a
CUDA error:

- Waiting for the GPU, e.g., by calling `synchronize`, copying data to CPU memory, or
  accessing an array element on the CPU. Captured operations only execute when the graph
  is launched, so their results aren't available while capturing.
- Creating library handles or plans, e.g., when using a library for the first time in a
  task. Execute the code once before capturing it.
- Using arrays that are allocated within a `GPUArrays.@cached` scope, as the cache reuses
  their memory after the scope ends.

There are also some operations that can be captured, but that behave differently when
launching the graph:

- Random numbers generated within kernels using `rand()`, or by the native random number
  generator (`CUDA.RNG`, as returned by `CUDA.default_rng()`), are the same every time the
  graph is launched. `rand!(A)` and `CUDA.rand`, which use cuRAND, generate new numbers
  for every launch.
- Graphs can only be launched on the device they were captured on.

Seeding a cuRAND generator synchronizes the device, which causes captures by other tasks to
fail ([#3336](https://github.com/JuliaGPU/CUDA.jl/issues/3336)). That happens when a task
uses cuRAND for the first time (e.g., `CUDA.rand`), or calls `Random.seed!` on a cuRAND
generator, while another task is capturing a graph.

Finally, graphs that are captured using the driver API directly, instead of using
[`capture`](@ref) or [`capture!`](@ref), aren't integrated with CUDA.jl's memory
management, and aren't guaranteed to keep the memory they use alive.

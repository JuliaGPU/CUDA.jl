# Workflow

A typical approach for porting or developing an application for the GPU is as follows:

1. develop an application using generic array functionality, and test it on the CPU with the
   `Array` type
2. port your application to the GPU by switching to the `CuArray` type
3. disallow the CPU fallback ("scalar indexing") to find operations that are not implemented
   for or incompatible with GPU execution
4. (optional) use lower-level, CUDA-specific interfaces to implement missing functionality
   or optimize performance


## [Scalar indexing](@id UsageWorkflowScalar)

Many array operations in Julia are implemented using loops, processing one element at a
time. Doing so with GPU arrays is very ineffective, as the loop won't actually execute on
the GPU, but transfer one element at a time and process it on the CPU. As this wrecks
performance, you will be warned when performing this kind of iteration:

```julia
julia> a = CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> a[1] += 1
┌ Warning: Performing scalar indexing.
│ ...
└ @ GPUArrays ~/Julia/pkg/GPUArrays/src/host/indexing.jl:57
2
```

Scalar indexing is only allowed in an interactive session, e.g. the REPL, because it is
convenient when porting CPU code to the GPU. If you want to disallow scalar indexing, e.g.
to verify that your application executes correctly on the GPU, call the `allowscalar`
function:

```julia
julia> CUDA.allowscalar(false)

julia> a[1] .+ 1
ERROR: scalar getindex is disallowed
Stacktrace:
 [1] error(::String) at ./error.jl:33
 [2] assertscalar(::String) at GPUArrays/src/indexing.jl:14
 [3] getindex(::CuArray{Int64,1,Nothing}, ::Int64) at GPUArrays/src/indexing.jl:54
 [4] top-level scope at REPL[5]:1

julia> a .+ 1
1-element CuArray{Int64,1,Nothing}:
 2
```

In a non-interactive session, e.g. when running code from a script or application, scalar
indexing is disallowed by default. There is no global toggle to allow scalar indexing; if
you really need it, you can mark expressions using `allowscalar` with do-block syntax or
`@allowscalar` macro:

```julia
julia> a = CuArray([1])
1-element CuArray{Int64, 1}:
 1

julia> CUDA.allowscalar(false)

julia> CUDA.allowscalar() do
         a[1] += 1
       end
2

julia> CUDA.@allowscalar a[1] += 1
3
```

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

To facilitate porting code, `CuArray` supports executing so-called "scalar code" which
processes one element at a time, e.g., in a for loop. Given how a GPU works, this is
extremely slow and will negate any performance benefit of using a GPU. As such, you will be
warned when performing this kind of iteration:

```julia
julia> a = CuArray([1])
1-element CuArray{Int64,1,Nothing}:
 1

julia> a[1] += 1
┌ Warning: Performing scalar operations on GPU arrays: This is very slow, consider disallowing these operations with `allowscalar(false)`
└ @ GPUArrays GPUArrays/src/indexing.jl:16
2
```

Once you've verified that your application executes correctly on the GPU, you should
disallow scalar indexing and use GPU-friendly array operations instead:

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

Many array operations however have been implemented themselves using scalar indexing. As a
result, calling into a seemingly GPU-friendly array operation might error out:

```julia
julia> a = CuArray([1,2])
2-element CuArray{Int64,1,Nothing}:
 1
 2

julia> var(a)
0.5

julia> var(a,dims=1)
ERROR: scalar getindex is disallowed
```

To resolve such issues, many array operations for `CuArray` are replaced with GPU-friendly
alternatives. If you run into a case like this, have a look at the CUDA.jl issue tracker and
file a bug report if there isn't one yet.

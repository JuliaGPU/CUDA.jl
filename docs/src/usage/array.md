# Array programming

The easiest way to use the GPU's massive parallelism, is by expressing operations in terms
of arrays: CUDA.jl provides an array type, `CuArray`, and many specialized array operations
that execute efficiently on the GPU hardware. In this section, we will briefly demonstrate
use of the `CuArray` type. Since we expose CUDA's functionality by implementing existing
Julia interfaces on the `CuArray` type, you should refer to the [upstream Julia
documentation](https://docs.julialang.org) for more information on these operations.

If you encounter missing functionality, or are running into operations that trigger
so-called ["scalar iteration"](@UsageWorkflowScalar), have a look at the [issue
tracker](https://github.com/JuliaGPU/CUDA.jl/issues) and file a new issue if there's none.
Do note that you can always access the underlying CUDA APIs by calling into the relevant
submodule. For example, if parts of the Random interface isn't properly implemented by
CUDA.jl, you can look at the CURAND documentation and possibly call methods from the
`CURAND` submodule directly. These submodules are available after importing the CUDA
package.


## Construction and Initialization

The `CuArray` type aims to implement the `AbstractArray` interface, and provide
implementations of methods that are commonly used when working with arrays. That means you
can construct `CuArray`s in the same way as regular `Array` objects:

```jldoctest
julia> CuArray{Int}(undef, 2)
2-element CuArray{Int64,1,Nothing}:
 0
 0

julia> CuArray{Int}(undef, (1,2))
1×2 CuArray{Int64,2,Nothing}:
 0  0

julia> similar(ans)
1×2 CuArray{Int64,2,Nothing}:
 0  0
```

Copying memory to or from the GPU can be expressed using constructors as well, or by calling
`copyto!`:

```jldoctest
julia> a = CuArray([1,2])
2-element CuArray{Int64,1,Nothing}:
 1
 2

julia> b = Array(a)
2-element Array{Int64,1}:
 1
 2

julia> copyto!(b, a)
2-element Array{Int64,1}:
 1
 2
```


## Higher-order abstractions

The real power of programming GPUs with arrays comes from Julia's higher-order array
abstractions: Operations that take user code as an argument, and specialize execution on it.
With these functions, you can often avoid having to write custom kernels. For example, to
perform simple element-wise operations you can use `map` or `broadcast`:

```jldoctest
julia> a = CuArray{Float32}(undef, (1,2));

julia> a .= 5
1×2 CuArray{Float32,2,Nothing}:
 5.0  5.0

julia> map(sin, a)
1×2 CuArray{Float32,2,Nothing}:
 -0.958924  -0.958924
```

To reduce the dimensionality of arrays, CUDA.jl implements the various flavours of
`(map)reduce(dim)`:

```jldoctest
julia> a = CUDA.ones(2,3)
2×3 CuArray{Float32,2,Nothing}:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> reduce(+, a)
6.0f0

julia> mapreduce(sin, *, a; dims=2)
2×1 CuArray{Float32,2,Nothing}:
 0.59582335
 0.59582335

julia> b = similar(a, 1)
1-element CuArray{Float32,1,Nothing}:
 6.0

julia> Base.mapreducedim!(identity, min, b, a)
1×1 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 1.0
```

To retain intermediate values, you can use `accumulate`:

```jldoctest
julia> a = CUDA.ones(2,3)
2×3 CuArray{Float32,2,Nothing}:
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> accumulate(+, a; dims=2)
2×3 CuArray{Float32,2,Nothing}:
 1.0  2.0  3.0
 1.0  2.0  3.0
```


## Logical operations

`CuArray`s can also be indexed with arrays of boolean values to select items:

```jldoctest
julia> a = CuArray([1,2,3])
3-element CuArray{Int64,1,Nothing}:
 1
 2
 3

julia> a[[false,true,false]]
1-element CuArray{Int64,1,Nothing}:
 2
```

Built on top of this, are several functions with higher-level semantics:

```jldoctest
julia> a = CuArray([11,12,13])
3-element CuArray{Int64,1,Nothing}:
 11
 12
 13

julia> findall(isodd, a)
2-element CuArray{Int64,1,Nothing}:
 1
 3

julia> findfirst(isodd, a)
1

julia> b = CuArray([11 12 13; 21 22 23])
2×3 CuArray{Int64,2,Nothing}:
 11  12  13
 21  22  23

julia> findmin(b)
(11, CartesianIndex(1, 1))

julia> findmax(b; dims=2)
([13; 23], CartesianIndex{2}[CartesianIndex(1, 3); CartesianIndex(2, 3)])
```


## Array wrappers

To some extent, CUDA.jl also supports well-known array wrappers from the standard library:

```jldoctest
julia> a = CuArray(collect(1:10))
10-element CuArray{Int64,1,Nothing}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10

julia> a = CuArray(collect(1:6))
6-element CuArray{Int64,1,Nothing}:
 1
 2
 3
 4
 5
 6

julia> b = reshape(a, (2,3))
2×3 CuArray{Int64,2,CuArray{Int64,1,Nothing}}:
 1  3  5
 2  4  6

julia> c = view(a, 2:5)
4-element CuArray{Int64,1,CuArray{Int64,1,Nothing}}:
 2
 3
 4
 5
```

The above contiguous `view` and `reshape` have been specialized to return new objects of
type `CuArray`. Other wrappers, such as non-contiguous views or the LinearAlgebra wrappers
that will be discussed below, are implemented using their own type (e.g. `SubArray` or
`Transpose`). This can cause problems, as calling methods with these wrapped objects will
not dispatch to specialized `CuArray` methods anymore. That may result in a call to fallback
functionality that performs scalar iteration.

Certain common operations, like broadcast or matrix multiplication, do know how to deal with
array wrappers by using the [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) package. This
is still not a complete solution though, e.g. new array wrappers are not covered, and only
one level of wrapping is supported. Sometimes the only solution is to materialize the
wrapper to a `CuArray` again.


## Random numbers

Base's convenience functions for generating random numbers are available in the CUDA module
as well:

```jldoctest
julia> CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.5278814
 0.86173964

julia> CUDA.randn(Float64, 2, 1)
2×1 CuArray{Float64,2,Nothing}:
 -0.7986343050328781
 -0.2333469420701086
```

Behind the scenes, these random numbers come from two different generators: one backed by
[CURAND](https://docs.nvidia.com/cuda/curand/index.html), another by kernels defined in
GPUArrays.jl. Operations on these generators are implemented using methods from the Random
standard library:

```jldoctest
julia> using Random

julia> a = Random.rand(CURAND.generator(), Float32, 1)
1-element CuArray{Float32,1,Nothing}:
 0.5068406

julia> using GPUArrays

julia> a = Random.rand!(GPUArrays.global_rng(a), a)
1-element CuArray{Float32,1,Nothing}:
 0.58538806
```

CURAND also supports generating lognormal and Poisson-distributed numbers:

```jldoctest
julia> CUDA.rand_logn(Float32, 1, 5; mean=2, stddev=20)
1×5 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 7.02912f-9  33.3227  0.278724  2.8658f13  4.24994f11

julia> CUDA.rand_poisson(UInt32, 1, 10; lambda=100)
1×10 CuArray{UInt32,2,Nothing}:
 0x00000067  0x0000006c  0x0000005d  0x00000065  0x00000065  0x00000063  0x0000005f  0x00000068  0x0000006a  0x0000006e
```

Note that these custom operations are only supported on a subset of types.


## Linear algebra

CUDA's linear algebra functionality from the [CUBLAS](https://developer.nvidia.com/cublas)
library is exposed by implementing methods in the LinearAlgebra standard library:

```jldoctest
julia> # enable logging to demonstrate a CUBLAS kernel is used
       CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)

julia> CUDA.rand(2,2) * CUDA.rand(2,2)
I! cuBLAS (v10.2) function cublasStatus_t cublasSgemm_v2(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int) called
2×2 CuArray{Float32,2,Nothing}:
 0.295727  0.479395
 0.624576  0.557361
```

Certain operations, like the above matrix-matrix multiplication, also have a native fallback
written in Julia for the purpose of working with types that are not supported by CUBLAS:

```jldoctest
julia> # enable logging to demonstrate no CUBLAS kernel is used
       CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)

julia> CUDA.rand(Int128, 2, 2) * CUDA.rand(Int128, 2, 2)
2×2 CuArray{Int128,2,Nothing}:
 151499160030096859457691185134444419729  -97014463222125585750517660033492187340
  24858862404898964861632634177015389670  121029362105248597343192066347336219384
```

Operations that exist in CUBLAS, but are not (yet) covered by high-level constructs in the
LinearAlgebra standard library, can be accessed directly from the CUBLAS submodule. Note
that you do not need to call the C wrappers directly (e.g. `cublasDdot`), as many operations
have more high-level wrappers available as well (e.g. `dot`):

```jldoctest
julia> x = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.2977523
 0.30158097

julia> y = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.5144331
 0.22614105

julia> CUBLAS.dot(2, x, 0, y, 0)
0.30634725f0

julia> dot(Array(x), Array(y))
0.22137347f0
```


## Solver

LAPACK-like functionality as found in the
[CUSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) library can be accessed through
methods in the LinearAlgebra standard library too:

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.283656  0.041456
 0.660603  0.509684

julia> a = a * a'
2×2 CuArray{Float32,2,Nothing}:
 0.0821795  0.208513
 0.208513   0.696174

julia> cholesky(a)
Cholesky{Float32,CuArray{Float32,2,Nothing}}
U factor:
2×2 UpperTriangular{Float32,CuArray{Float32,2,Nothing}}:
 0.28667  0.727365
  ⋅       0.408795
```

Other operations are bound to the left-division operator:

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.531502  0.00622418
 0.169554  0.223502

julia> b = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.883509  0.547314
 0.756986  0.486571

julia> a \ b
2×2 CuArray{Float32,2,Nothing}:
 1.63717  1.01326
 2.14494  1.40835

julia> Array(a) \ Array(b)
2×2 Array{Float32,2}:
 1.63717  1.01326
 2.14494  1.40835
```



## Sparse arrays

Sparse array functionality from the
[CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) library is mainly available
through functionality from the SparseArrays package applied to `CuSparseArray` objects:

```jldoctest
julia> using SparseArrays

julia> x = sprand(10,0.2)
10-element SparseVector{Float64,Int64} with 4 stored entries:
  [1 ]  =  0.823306
  [2 ]  =  0.402525
  [6 ]  =  0.352595
  [10]  =  0.475461

julia> using CUDA.CUSPARSE

julia> d_x = CuSparseVector(x)
10-element CuSparseVector{Float64} with 4 stored entries:
  [1 ]  =  0.823306
  [2 ]  =  0.402525
  [6 ]  =  0.352595
  [10]  =  0.475461

julia> nonzeros(d_x)
4-element CuArray{Float64,1,Nothing}:
 0.8233063097156732
 0.4025250793787798
 0.35259544625232353
 0.4754608776715703

julia> nnz(d_x)
4
```

For 2-D arrays the `CuSparseMatrixCSC` and `CuSparseMatrixCSR` can be used.

Non-integrated functionality can be access directly in the CUSPARSE submodule again.


## FFTs

Functionality from [CUFFT](https://docs.nvidia.com/cuda/cufft/index.html) is integrated with
the interfaces from the [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl)
package. You can use them by importing the FFTW package:

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.526821  0.972028
 0.152439  0.23469

julia> using FFTW

julia> fft(a)
2×2 CuArray{Complex{Float32},2,Nothing}:
 1.88598+0.0im  -0.527457+0.0im
 1.11172+0.0im  -0.362956+0.0im
```

# Array programming

```@meta
DocTestSetup = quote
    using CUDA

    import Random
    Random.seed!(0)

    CURAND.seed!(0)
end
```

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

```julia
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

julia> b = CUDA.zeros(1)
1-element CuArray{Float32,1,Nothing}:
 0.0

julia> Base.mapreducedim!(identity, +, b, a)
1×1 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 6.0
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
 0.74021935
 0.9209938

julia> CUDA.randn(Float64, 2, 1)
2×1 CuArray{Float64,2,Nothing}:
 -0.3893830994647195
  1.618410515635752
```

Behind the scenes, these random numbers come from two different generators: one backed by
[CURAND](https://docs.nvidia.com/cuda/curand/index.html), another by kernels defined in
GPUArrays.jl. Operations on these generators are implemented using methods from the Random
standard library:

```jldoctest
julia> using Random

julia> a = Random.rand(CURAND.generator(), Float32, 1)
1-element CuArray{Float32,1,Nothing}:
 0.74021935

julia> using GPUArrays

julia> a = Random.rand!(GPUArrays.global_rng(a), a)
1-element CuArray{Float32,1,Nothing}:
 0.13394515
```

CURAND also supports generating lognormal and Poisson-distributed numbers:

```jldoctest
julia> CUDA.rand_logn(Float32, 1, 5; mean=2, stddev=20)
1×5 CuArray{Float32,2,CuArray{Float32,1,Nothing}}:
 2567.61  4.256f-6  54.5948  0.00283999  9.81175f22

julia> CUDA.rand_poisson(UInt32, 1, 10; lambda=100)
1×10 CuArray{UInt32,2,Nothing}:
 0x00000058  0x00000066  0x00000061  …  0x0000006b  0x0000005f  0x00000069
```

Note that these custom operations are only supported on a subset of types.


## Linear algebra

CUDA's linear algebra functionality from the [CUBLAS](https://developer.nvidia.com/cublas)
library is exposed by implementing methods in the LinearAlgebra standard library:

```julia
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

```julia
julia> # enable logging to demonstrate no CUBLAS kernel is used
       CUBLAS.cublasLoggerConfigure(1, 0, 1, C_NULL)

julia> CUDA.rand(Int128, 2, 2) * CUDA.rand(Int128, 2, 2)
2×2 CuArray{Int128,2,Nothing}:
 -147256259324085278916026657445395486093  -62954140705285875940311066889684981211
 -154405209690443624360811355271386638733  -77891631198498491666867579047988353207
```

Operations that exist in CUBLAS, but are not (yet) covered by high-level constructs in the
LinearAlgebra standard library, can be accessed directly from the CUBLAS submodule. Note
that you do not need to call the C wrappers directly (e.g. `cublasDdot`), as many operations
have more high-level wrappers available as well (e.g. `dot`):

```jldoctest
julia> x = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.74021935
 0.9209938

julia> y = CUDA.rand(2)
2-element CuArray{Float32,1,Nothing}:
 0.03902049
 0.9689629

julia> CUBLAS.dot(2, x, 0, y, 0)
0.057767443f0

julia> using LinearAlgebra

julia> dot(Array(x), Array(y))
0.92129254f0
```


## Solver

LAPACK-like functionality as found in the
[CUSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) library can be accessed through
methods in the LinearAlgebra standard library too:

```jldoctest
julia> using LinearAlgebra

julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.740219  0.0390205
 0.920994  0.968963

julia> a = a * a'
2×2 CuArray{Float32,2,Nothing}:
 0.549447  0.719547
 0.719547  1.78712

julia> cholesky(a)
Cholesky{Float32,CuArray{Float32,2,Nothing}}
U factor:
2×2 UpperTriangular{Float32,CuArray{Float32,2,Nothing}}:
 0.741247  0.970725
  ⋅        0.919137
```

Other operations are bound to the left-division operator:

```jldoctest
julia> a = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.740219  0.0390205
 0.920994  0.968963

julia> b = CUDA.rand(2,2)
2×2 CuArray{Float32,2,Nothing}:
 0.925141  0.667319
 0.44635   0.109931

julia> a \ b
2×2 CuArray{Float32,2,Nothing}:
  1.29018    0.942772
 -0.765663  -0.782648

julia> Array(a) \ Array(b)
2×2 Array{Float32,2}:
  1.29018    0.942773
 -0.765663  -0.782648
```



## Sparse arrays

Sparse array functionality from the
[CUSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) library is mainly available
through functionality from the SparseArrays package applied to `CuSparseArray` objects:

```jldoctest
julia> using SparseArrays

julia> x = sprand(10,0.2)
10-element SparseVector{Float64,Int64} with 4 stored entries:
  [3 ]  =  0.585812
  [4 ]  =  0.539289
  [7 ]  =  0.260036
  [8 ]  =  0.910047

julia> using CUDA.CUSPARSE

julia> d_x = CuSparseVector(x)
10-element CuSparseVector{Float64} with 4 stored entries:
  [3 ]  =  0.585812
  [4 ]  =  0.539289
  [7 ]  =  0.260036
  [8 ]  =  0.910047

julia> nonzeros(d_x)
4-element CuArray{Float64,1,Nothing}:
 0.5858115517433242
 0.5392892841426182
 0.26003585026904785
 0.910046541351011

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
 0.740219  0.0390205
 0.920994  0.968963

julia> using FFTW

julia> fft(a)
2×2 CuArray{Complex{Float32},2,Nothing}:
   2.6692+0.0im   0.65323+0.0im
 -1.11072+0.0im  0.749168+0.0im
```

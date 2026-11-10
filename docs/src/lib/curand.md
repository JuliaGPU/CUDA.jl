# cuRAND

```@meta
CurrentModule = cuRAND
```

cuRAND.jl generates random numbers on the GPU, with NVIDIA's cuRAND library or with
kernels written in Julia. It offers them at three levels:

- Random's in-place functions `rand!` and `randn!` fill `CuArray`s, and Random's `rand`,
  `randn`, `rand!`, `randn!` and `seed!` work with the generators below:
  [`LibraryRNG`](@ref), backed by cuRAND, and [`NativeRNG`](@ref), which calls the
  device-side `rand` in a kernel.
- [`cuRAND.rand`](@ref) and [`cuRAND.randn`](@ref), also available as `CUDA.rand` and
  `CUDA.randn`, generate `CuArray`s with the current task's generators, as do
  [`rand_logn`](@ref) and [`rand_poisson`](@ref) for distributions that Random does not
  offer. [`cuRAND.seed!`](@ref) (`CUDA.seed!`) seeds those generators. These are different
  functions than `Random.rand` etc., which work on the CPU, so call them qualified.
- The generated bindings in `libcurand.jl` follow NVIDIA's C API, and change with the CUDA
  toolkit. A [`LibraryRNG`](@ref) converts to the `curandGenerator_t` they take.

```julia
using CUDA, cuRAND, Random

A = cuRAND.rand(Float32, 1024)      # uniform on (0, 1]
B = cuRAND.randn(1024; stddev=2)    # normal, Float32 by default
randn!(B)                           # in place

rng = cuRAND.LibraryRNG()
Random.seed!(rng, 42)
C = rand(rng, Float64, 16)          # a CuArray, using this generator
```

Each task has its own generators, one per device: [`library_rng`](@ref) for the types
cuRAND supports, [`gpuarrays_rng`](@ref) for the others, and [`native_rng`](@ref).

## Public

```@autodocs
Modules = [cuRAND]
Private = false
```

## Private

```@autodocs
Modules = [cuRAND]
Public = false
```

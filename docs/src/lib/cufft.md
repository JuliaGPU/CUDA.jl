# cuFFT

```@meta
CurrentModule = cuFFT
```

cuFFT.jl offers NVIDIA's fast Fourier transforms at two levels:

- The functions of [AbstractFFTs.jl](https://github.com/JuliaMath/AbstractFFTs.jl), which
  cuFFT.jl re-exports, work on `CuArray`s: `fft`, `ifft`, `bfft`, `rfft`, `irfft`,
  `brfft`, the in-place `fft!`, `ifft!` and `bfft!`, and the planning functions
  `plan_fft` etc. that return a reusable [`CuFFTPlan`](@ref), or an
  `AbstractFFTs.ScaledPlan` wrapping one. cuFFT transforms `Float16`, `Float32` and
  `Float64` data and their complex versions; integer and rational inputs are converted to
  floating point first. Strided views are supported, and are copied when cuFFT cannot
  address them directly.
- The generated bindings in `libcufft.jl` follow NVIDIA's C API, and change with the CUDA
  toolkit. A `CuFFTPlan` converts to the `cufftHandle` they take.

```julia
using CUDA, cuFFT

x = CUDA.rand(ComplexF32, 1024, 64)
p = plan_fft(x, 1)      # transform the first dimension, batched over the second
y = p * x
p \ y ≈ x               # true
```

## Public

```@autodocs
Modules = [cuFFT]
Private = false
```

## Private

```@autodocs
Modules = [cuFFT]
Public = false
```

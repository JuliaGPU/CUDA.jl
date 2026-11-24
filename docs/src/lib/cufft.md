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

## Fusing broadcasts into transforms

Element-wise work around a transform, such as windowing, converting or scaling, normally
runs as separate kernels that each pass over the data. With the `@~` macro of
[LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl), cuFFT executes it as part of
the transform instead:

```julia
using CUDA, cuFFT, LazyArrays

p = plan_fft(x, 1)
y = abs2.(@~ p * (w .* x))      # window, transform and take the power in one go
y .= (@~ p * (w .* x)) ./ n     # store into an existing array
acc .+= abs2.(@~ p * x)         # accumulate; the destination can be part of the broadcast
```

`@~ p * arg` is a lazy `p * arg`, where `arg` is an array or a broadcast. cuFFT fuses that
broadcast, and the broadcast the result is used in, into the transform. The result is the
same as without `@~`, up to rounding. The input can have any element type that converts
to the plan's, e.g. integer samples, and a plan that normalizes (`plan_ifft`, `inv(p)`)
does so as part of the transform.

`@~` makes every call in its expression lazy, so give it variables: `@~ p * x` rather
than `@~ plan_fft(x) * x`.

Fusion needs CUDA 13.5 or newer and a GPU with compute capability 10.0 (Blackwell) or
newer, and applies to out-of-place `Float32` and `Float64` transforms. In all other cases
the same code still works: cuFFT then executes `p * arg` and the broadcast around it
separately. That also happens when:

- the transform has a size with a prime factor of 127 or more, a layout that cuFFT has to
  permute, or is one of the few for which cuFFT doesn't support fusion;
- the broadcast may throw an exception, e.g. because it converts to an integer;
- the broadcast contains several transforms, or changes the shape of the output;
- the destination is not a dense `CuArray`, or shares memory with an input;
- the code is captured into a graph.

Complex-to-real transforms only fuse the broadcast around them. To find out why a
broadcast is not fused, enable debug logging with `JULIA_DEBUG=cuFFT`.

Fused broadcasts are compiled and linked into cuFFT's kernels the first time they are
used, which takes longer than compiling an ordinary broadcast.

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

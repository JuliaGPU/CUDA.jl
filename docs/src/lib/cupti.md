# CUPTI

```@meta
CurrentModule = CUPTI
```

CUPTI, the CUDA Profiling Tools Interface, is what CUDA.jl's profilers are built on. Most
users should use those instead: `CUDA.@profile` traces API calls and kernels,
`CUDA.@metrics` collects hardware metrics for each kernel, and `CUDA.@sample` samples
where kernels spend their time.

The module offers CUPTI at two levels:

- The functions documented below wrap CUPTI's APIs for tools: callbacks on API calls and
  other events ([`CallbackConfig`](@ref)), tracing of activity such as kernel launches
  and memory copies ([`ActivityConfig`](@ref)), hardware metrics per kernel
  ([`range_profile`](@ref)) and PC sampling ([`pc_sample`](@ref)). They are not exported,
  so call them as `CUPTI.range_profile(...)`.
- The generated bindings in `libcupti.jl` follow NVIDIA's C API, and change with the CUDA
  toolkit. The records [`process`](@ref) passes and the enum values the configurations
  take come from there.

CUPTI reports two versions. [`CUPTI.version`](@ref) is the version of its API, which looks
like a CUDA version and only changes when the API does, so check it for API features.
[`CUPTI.library_version`](@ref) is the release of the library, which is numbered by year
(e.g. `v"2026.4.0"`) and changes with every CUDA update release, so it identifies a library
with a known bug.

Profiling may need permissions: see [`can_profile`](@ref), and NVIDIA's documentation on
`ERR_NVGPUCTRPERM` for hardware metrics and PC sampling. A failing CUPTI call throws a
[`CUPTIError`](@ref).

## Public

```@autodocs
Modules = [CUPTI]
Private = false
```

## Private

```@autodocs
Modules = [CUPTI]
Public = false
```

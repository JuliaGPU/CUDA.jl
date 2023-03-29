# Wrapping headers

This directory contains scripts that can be used to automatically generate
wrappers for C headers by NVIDIA, such as CUBLAS or CUDNN. This is done using
Clang.jl.

In CUDA.jl, the wrappers need to know whether pointers passed into the
library point to CPU or GPU memory (i.e. `Ptr` or `CuPtr`). This information is
not available from the headers, and instead should be provided by the developer. The
specific information is embedded in the TOML files for each component.


# Usage

Either run `wrap.jl` directly, or include it using Revise.jl and call the `main()` function.
Be sure to activate the project environment in this folder, which will download CUDA from
artifacts (if you want to upgrade the headers, be sure to update the relevant JLLs in the
project environment).
You can also call `main(library)` if you want to generate the wrapper for a specific `library`.
The possible values for `library` are `"all"` (default), `"cuda"`, `"nvml"`, `"cupti"`,
`"cublas"`, `"cufft"`, `"curand"`, `"cusparse"`, `"cusolver"`, `"cudnn"`, `"cutensor"`,
`"nvperf_host"`, `"cutensornet"` or `"custatevec"`.

For each library, the script performs the following steps:

- generate wrappers with Clang.jl
- rewrite the headers: wrap functions that result status codes with `@check`, add calls to
  API initializers, etc.

You should always review any changes to the headers! Specifically, verify that pointer
arguments are of the correct type, and if they aren't, modify the TOML file and regenerate
the wrappers. The following types should be considered as alternatives to plain `Ptr`s:

- `CuPtr`: if this pointer is a device pointer
- `PtrOrCuPtr`: if this pointer can be either a device or host pointer
- `Ref`: if the pointer represents a scalar or single-value argument on the host
- `CuRef`: idem, but on the device
- `RefOrCuRef`: idem, but either on the host or device

# Wrapping headers

This directory contains scripts that can be used to automatically generate
wrappers for C headers by NVIDIA, such as CUBLAS or CUDNN. This is done using
Clang.jl, with some CSTParser.jl-based scripts to clean-up the result.

In CUDA.jl, the wrappers need to know whether pointers passed into the
library point to CPU or GPU memory (i.e. `Ptr` or `CuPtr`). This information is
not available from the headers, and the headers will need to be reviewed up manually.



# Usage

Either run `wrap.jl` directly, or include it using Revise.jl and call the `main()` function.
Be sure to activate the project environment in this folder, which will download CUDA from
artifacts (if you want to upgrade the headers, be sure to update the relevant JLLs in the
project environment).
You can also call `main(name=library)` if you want to generate the wrapper for a specific `library`.
The possible values for `library` are `"all"` (default), `"cudadrv"`, `"nvtx"`, `"nvml"`,
`"cupti"`, `"cublas"`, `"cufft"`, `"curand"`, `"cusparse"`, `"cusolver"`, `"cudnn"`, `"cutensor"`,
`"cutensornet"` or `"custatevec"`.

For each library, the script performs the following steps:

- generate wrappers with Clang.jl
- rewrite the headers: wrap functions that result status codes with `@check`, add calls to
  API initializers, etc.
- apply manual patches: these are read from the `patches` folder, and can be used to
  incompatible code

Clang.jl generates headers with two files: a `common` file with type definitions, aliases,
etc, and a main wrapper that contains function definitions. The former will be copied over
the existing files automatically, while for the latter we scan for changes: Removed
functions are put in the `libXXX_deprecated.jl` file, new ones are concatenated to the
`libXXX.jl` file.

You should always review any changes to the headers! Specifically, to correct `Ptr`
signature and possibly change them to:
- `CuPtr`: if this pointer is a device pointer
- `PtrOrCuPtr`: if this pointer can be either a device or host pointer
- `Ref`: if the pointer represents a scalar or single-value argument on the host
- `CuRef`: idem, but on the device
- `RefOrCuRef`: idem, but either on the host or device

Finally, it might also be useful to diff the generated wrapper (generated from scratch) in
the `res/wrap` folder with the one in the `lib` folder (which is incrementally changed) to
see if no function signatures have changed.

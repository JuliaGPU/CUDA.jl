# [Overview](@id InstallationOverview)

The Julia CUDA stack only requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx). It is not necessary to install the
[CUDA toolkit](https://developer.nvidia.com/cuda-downloads). On Windows, also make sure you
have the [Visual C++ redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe)
installed.



## Package installation

For most users, installing the latest tagged version of CUDA.jl will be sufficient. You can
easily do that using the package manager:

```
pkg> add CUDA
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("CUDA")
```

In some cases, you might need to use the `main` version of this package, e.g., because it
includes a specific fix you need. Often, however, the development version of this package
itself relies on unreleased versions of other packages. This information is recorded in the
manifest at the root of the repository, which you can use by starting Julia from the CUDA.jl
directory with the `--project` flag:

```
$ cd .julia/dev/CUDA.jl     # or wherever you have CUDA.jl checked out
$ julia --project
pkg> instantiate            # to install correct dependencies
julia> using CUDA
```

In the case you want to use the development version of CUDA.jl with other packages, you
cannot use the manifest and you need to manually install those dependencies from the master
branch. Again, the exact requirements are recorded in CUDA.jl's manifest, but often the
following instructions will work:

```
pkg> add GPUCompiler#master
pkg> add GPUArrays#master
pkg> add LLVM#master
```



## Platform support

We support the same operating systems that NVIDIA supports: Linux and Windows, on x86_64
and (on Linux) aarch64. The main development platform, and the only CI system, is x86_64
Linux, so on other combinations there might be bugs. Refer to the [support
matrix](https://github.com/JuliaGPU/CUDA.jl#requirements) in the README for the level of
support per platform, CUDA version and device architecture.

### NVIDIA Jetson

Tegra-based systems (the Jetson series) are supported, and `Pkg.add("CUDA")` selects the
`cuda_platform=jetson` toolkit builds automatically. What you end up with depends on the
JetPack generation the board runs:

| Board (JetPack, L4T) | System CUDA | CUDA.jl uses |
|---|---|---|
| Jetson Nano, TX1, TX2 (JetPack 4, r32) | 10.2 | system driver, CUDA 10.2 artifacts |
| Xavier (JetPack 5, r35) | 11.4 | bundled CUDA 12.2 L4T driver, CUDA 12.5 artifacts |
| Orin (JetPack 6, r36) | 12.x | bundled CUDA 12.9 L4T driver, CUDA 12.9 artifacts (not validated on hardware) |
| Orin, Thor (JetPack 7, r39) | 13.x | system driver, CUDA 13.x artifacts |

On JetPack 5 and 6, `CUDA_Driver_jll` bundles NVIDIA's L4T forward-compatibility driver
for that kernel-mode driver generation, and only loads it after verifying that it
initializes and supports every device present. Set the `compat` preference on
`CUDA_Driver_jll` to `false` (or `JULIA_CUDA_USE_COMPAT=false`) to keep the system driver.

Two Jetson-specific caveats:

- NVIDIA's Jetson library redistributables only contain device code for the architectures
  of the JetPack generation they belong to. Xavier (sm_72) code disappears from cuBLAS in
  CUDA 12.6 and from all libraries in 12.8, so on Xavier the toolkit selection is capped
  at CUDA 12.5 even though `ptxas` would still target sm_72. Setting an explicit `version`
  preference bypasses that cap, and the vendor libraries will then fail with errors such
  as `CUBLAS_STATUS_ARCH_MISMATCH`.
- Profiling and SASS reflection need extra permissions, see [Profiling on Tegra](@ref).



## NVIDIA driver

To use the Julia GPU stack, you need to install the NVIDIA driver for your system and GPU.
You can find [detailed instructions](https://www.nvidia.com/Download/index.aspx) on the
NVIDIA home page.

If you're using Linux you should always consider installing the driver through the package
manager of your distribution. In the case that driver is out of date or does not support
your GPU, and you need to download a driver from the NVIDIA home page, similarly prefer a
distribution-specific package (e.g., deb, rpm) instead of the generic runfile option.

If you are using a shared system, ask your system administrator on how to install or load
the NVIDIA driver. Generally, you should be able to find and use the CUDA driver library,
called `libcuda.so` on Linux, `libcuda.dylib` on macOS and `nvcuda64.dll` on Windows. You
should also be able to execute the `nvidia-smi` command, which lists all available GPUs you
have access to.

On some enterprise systems, CUDA.jl will be able to upgrade the driver for the duration of
the session (using CUDA's Forward Compatibility mechanism). This will be mentioned in the
`CUDA.versioninfo()` output, so be sure to verify that before asking your system
administrator to upgrade:

```
julia> CUDA.versioninfo()
CUDA toolchain:
- runtime 12.5.0, artifact installation
- driver 535.171.4 for CUDA 12.2, forward-compatible
- compiler 12.5.82, artifact installation
...
```

The same mechanism is used on JetPack 5 and 6 Jetson boards, see
[Platform support](@ref) above.

Finally, to be able to use all of the Julia GPU stack you need to have permission to profile
GPU code. On Linux, that means loading the `nvidia` kernel module with the
`NVreg_RestrictProfilingToAdminUsers=0` option configured (e.g., in `/etc/modprobe.d`).
Refer to the [following
document](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)
for more information.



## CUDA toolkit

The recommended way to use CUDA.jl is to let it automatically download an appropriate CUDA
toolkit. CUDA.jl will check your driver's capabilities, which versions of CUDA are available
for your platform, and automatically download an appropriate artifact containing all the
libraries that CUDA.jl supports.

If you *really* need to use a different CUDA toolkit, it's possible (but not recommended) to
load a different version, or even an installation from your local system. The runtime and
compiler are configured using `version` and `local` preferences on CUDA\_Runtime\_jll and
CUDA\_Compiler\_jll, but there is also a user-friendly API available in CUDA.jl.

### Specifying the CUDA version

You can choose which version to (try to) download and use by calling
`CUDA.set_runtime_version!`:

```
julia> using CUDA

julia> CUDA.set_runtime_version!(v"11.8")
[ Info: Configure the active project to use CUDA 11.8 from artifact sources; please re-start Julia for this to take effect.
```

This configures matching runtime and compiler artifacts. The preference is compatible with
other CUDA JLLs too: for example, `CUDNN_jll` will only select artifacts compatible with the
configured CUDA runtime.

`CUDA.set_runtime_version!` performs the following two actions:

1. Generates the following `LocalPreferences.toml` file in your active environment:

   ```
   [CUDA_Runtime_jll]
   version = "11.8"

   [CUDA_Compiler_jll]
   version = "11.8"
   ```

2. Ensures the following lines are in the `[extras]` section of your active `Project.toml`
   file so Preferences.jl can find both preferences:

   ```
   CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
   CUDA_Compiler_jll = "d1e2174e-dfdc-576e-b43e-73b79eb1aca8"
   ```

The JLL preferences can also be set independently, which is useful for testing compiler
releases while keeping the runtime fixed. Keep them within the same CUDA major because
runtime libraries link against compiler libraries with a major-versioned soname; a
mismatch shows up as a load failure while precompiling, e.g.

```
ERROR: LoadError: InitError: could not load library ".../lib/libcusolver.so"
libnvJitLink.so.12: cannot open shared object file: No such file or directory
```

Setting only one of the two preferences by hand has the same effect, because the other
JLL keeps the selection it cached earlier. `CUDA.set_runtime_version!` configures both,
and is the recommended way to change them.

CUDA 10.2 and 11 are supported on a best-effort basis: they are not covered by CI, and
not all functionality is available. Runtime and compiler artifacts are available for
every CUDA version CUDA.jl supports on Linux (x86_64 and Tegra aarch64) and Windows; on
other platforms, or for a version without artifacts, use a local toolkit as described
below.

### Using a local CUDA

To use a local installation, set the `local_toolkit` keyword argument to
`CUDA.set_runtime_version!`:

```
julia> using CUDA

julia> CUDA.versioninfo()
CUDA toolchain:
- runtime 11.8.0, artifact installation
...

julia> CUDA.set_runtime_version!(local_toolkit=true)
[ Info: Configure the active project to use the default CUDA from the local system; please re-start Julia for this to take effect.
```

Calling `CUDA.set_runtime_version!(local_toolkit=true)` generates the following `LocalPreferences.toml` file in
your active environment:

```
[CUDA_Runtime_jll]
local = "true"

[CUDA_Compiler_jll]
local = "true"
```

so that after re-launching Julia:

```
julia> using CUDA

julia> CUDA.versioninfo()
CUDA toolchain:
- runtime 11.8.0, local installation
...
```

The `local = "true"` preferences configure CUDA.jl to use discovery for the local runtime
and compiler and prevent downloading their artifacts. It may be useful to set them before
ever importing CUDA.jl, for example in a system-wide depot.

If CUDA.jl doesn't properly detect your local toolkit, it may be that certain libraries or
binaries aren't on a globally-discoverable path. For more information, run Julia with the
`JULIA_DEBUG` environment variable set to `CUDA_Runtime_Discovery`.

Note that using a local toolkit affects CUDA-related JLLs beyond CUDA\_Runtime\_jll and
CUDA\_Compiler\_jll. Packages that depend on those JLLs need to use
CUDA\_Runtime\_Discovery when the corresponding local preference is set.


## Precompiling CUDA.jl without CUDA

CUDA.jl can be precompiled and imported on systems without a GPU or CUDA installation. This
simplifies the situation where an application optionally uses CUDA. However, when CUDA.jl
is precompiled in such an environment, it *cannot* be used to run GPU code. This is a
result of artifacts being selected at precompile time: without a CUDA driver, neither the
CUDA runtime nor the CUDA compiler artifacts are selected, so nothing is downloaded, and
kernels cannot be compiled.

In some cases, e.g. with containers or HPC log-in nodes, you may want to precompile CUDA.jl
on a system without CUDA, yet still want to have it download the necessary artifacts and/or
produce a precompilation image that can be used on a system with CUDA. This can be achieved
by informing CUDA.jl which CUDA toolkit to run time by calling `CUDA.set_runtime_version!`.

When using artifacts, that's as simple as e.g. calling `CUDA.set_runtime_version!(v"11.8")`,
and afterwards re-starting Julia and re-importing CUDA.jl in order to trigger precompilation
again and download the necessary artifacts. If you want to use a local CUDA installation,
you also need to set the `local_toolkit` keyword argument, e.g., by calling
`CUDA.set_runtime_version!(v"11.8"; local_toolkit=true)`. Note that the version specified
here needs to match what will be available at run time. In both cases, i.e. when using
artifacts or a local toolkit, the chosen version needs to be compatible with the available
driver.

The same caching applies in reverse: a session where the GPU is hidden (say,
`CUDA_VISIBLE_DEVICES=""`) selects a toolkit sized on the driver version alone, and that
selection stays in the JLL's precompilation cache afterwards, because nothing about the
project changed. This is visible on a Jetson that would otherwise adopt a
forward-compatibility driver: it keeps the smaller, system-driver-sized toolkit. Setting
or clearing a preference re-triggers selection, as does recompiling the JLLs by hand:

```julia
for (uuid, name) in (("76a88914-d11a-5bdc-97e0-2f5a05c973a2", "CUDA_Runtime_jll"),
                     ("d1e2174e-dfdc-576e-b43e-73b79eb1aca8", "CUDA_Compiler_jll"))
    Base.compilecache(Base.PkgId(Base.UUID(uuid), name))
end
```

Finally, in such a scenario you may also want to call `CUDA.precompile_runtime()` to ensure
that the GPUCompiler runtime library is precompiled as well. This and all of the above is
demonstrated in the Dockerfile that's part of the CUDA.jl repository.

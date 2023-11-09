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

In some cases, you might need to use the `master` version of this package, e.g., because it
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

We support the same operation systems that NVIDIA supports: Linux, and Windows. Similarly,
we support x86, ARM, PPC, ... as long as Julia is supported on it and there exists an NVIDIA
driver and CUDA toolkit for your platform. The main development platform (and the only CI
system) however is x86_64 on Linux, so if you are using a more exotic combination there
might be bugs.



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
CUDA runtime 10.2
CUDA driver 11.8
NVIDIA driver 520.56.6, originally for CUDA 11.7
```

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
load a different version of the CUDA runtime, or even an installation from your local
system. Both are configured by setting the `version` preference (using Preferences.jl) on
the CUDA_Runtime_jll.jl package, but there is also a user-friendly API available in CUDA.jl.

### Specifying the CUDA version

You can choose which version to (try to) download and use by calling
`CUDA.set_runtime_version!`:

```
julia> using CUDA

julia> CUDA.set_runtime_version!(v"11.8")
[ Info: Set CUDA.jl toolkit preference to use CUDA 11.8.0 from artifact sources, please re-start Julia for this to take effect.
```

This generates the following `LocalPreferences.toml` file in your active environment:

```
[CUDA_Runtime_jll]
version = "11.8"
```

This preference is compatible with other CUDA JLLs, e.g., if you load `CUDNN_jll` it will
only select artifacts that are compatible with the configured CUDA runtime.

### Using a local CUDA

To use a local installation, you set the `local_toolkit` keyword argument to
`CUDA.set_runtime_version!`:

```
julia> using CUDA

julia> CUDA.versioninfo()
CUDA runtime 11.8, artifact installation
...

julia> CUDA.set_runtime_version!(local_toolkit=true)
[ Info: Set CUDA.jl toolkit preference to use CUDA from the local system, please re-start Julia for this to take effect.
```

After re-launching Julia:

```
julia> using CUDA

julia> CUDA.versioninfo()
CUDA runtime 11.8, local installation
...
```

Calling the above helper function generates the following `LocalPreferences.toml` file in
your active environment:

```
[CUDA_Runtime_jll]
local = "true"
```

This preference not only configures CUDA.jl to use a local toolkit, it also prevents
downloading any artifact, so it may be interesting to set this preference before ever
importing CUDA.jl (e.g., by putting this preference file in a system-wide depot).

If CUDA.jl doesn't properly detect your local toolkit, it may be that certain libraries or
binaries aren't on a globally-discoverable path. For more information, run Julia with the
`JULIA_DEBUG` environment variable set to `CUDA_Runtime_Discovery`.

Note that using a local toolkit instead of artifacts *any* CUDA-related JLL, not just of
`CUDA_Runtime_jll`. Any package that depends on such a JLL needs to inspect
`CUDA.local_toolkit`, and if set use `CUDA_Runtime_Discovery` to detect libraries and
binaries instead.


## Precompiling CUDA.jl without CUDA

CUDA.jl can be precompiled and imported on systems without a GPU or CUDA installation. This
simplifies the situation where an application optionally uses CUDA. However, when CUDA.jl
is precompiled in such an environment, it *cannot* be used to run GPU code. This is a
result of artifacts being selected at precompile time.

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

Finally, in such a scenario you may also want to call `CUDA.precompile_runtime()` to ensure
that the GPUCompiler runtime library is precompiled as well. This and all of the above is
demonstrated in the Dockerfile that's part of the CUDA.jl repository.

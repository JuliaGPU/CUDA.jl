# [Overview](@id InstallationOverview)

The Julia CUDA stack requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and corresponding [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). The former should be installed by you
or your system administrator, while the latter can be automatically downloaded by Julia
using the artifact subsystem.



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
┌ Warning: CUDA Runtime version set to 11.8, please re-start Julia for this to take effect.
└ @ CUDA /usr/local/share/julia/packages/CUDA/irdEw/lib/cudadrv/version.jl:54
```

This generates the following `LocalPreferences.toml` file in your active environment:

```
[CUDA_Runtime_jll]
version = "11.8"
```

This preference is compatible with other CUDA JLLs, e.g., if you load CUDNN_jll it will
only select artifacts that are compatible with the configured CUDA runtime.

### Using a local CUDA

To use a local installation, you can invoke the same API but set the version to `"local"`:

```
julia> using CUDA

julia> CUDA.versioninfo()
CUDA runtime 11.8, artifact installation
...

julia> CUDA.set_runtime_version!("local")
┌ Warning: CUDA Runtime version set to local, please re-start Julia for this to take effect.
└ @ CUDA ~/Julia/pkg/CUDA/lib/cudadrv/version.jl:73
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
version = "local"
```

This preference not only configures CUDA.jl to use a local toolkit, it also prevents
downloading any artifact, so it may be interesting to set this preference before ever
importing CUDA.jl (e.g., by putting this preference file in a system-wide depot).

If CUDA.jl doesn't properly detect your local toolkit, it may be that certain libraries or
binaries aren't on a globally-discoverable path. For more information, run Julia with the
`JULIA_DEBUG` environment variable set to `CUDA_Runtime_Discovery`.

Note that setting the version to `"local"` disables use of *any* CUDA-related JLL, not just
of CUDA_Runtime_jll. This is out of necessity: JLLs are baked in the precompilation image at
compile time, while local toolkit discovery happens at run time; this inconsistency makes it
impossible to select a compatible artifact for the JLLs. If you care about other JLLs, use
CUDA from artifacts.


## Containers

CUDA.jl is container friendly: You can install, precompile, and even import the package on a
system without a GPU:

```
$ docker run --rm -it julia   # note how we're *not* using `--gpus=all` here,
                              # so we won't have access to a GPU (or its driver)

pkg> add CUDA

pkg> precompile
Precompiling project...
[ Info: Precompiling CUDA [052768ef-5323-5732-b1bb-66c8b64840ba]
```

The above is common when building a container (`docker build` does not take a `--gpus`
argument). It does prevent CUDA.jl from downloading the toolkit artifacts that will be
required at run time, because it cannot query the driver for the CUDA compatibility level.

To avoid having to download the CUDA toolkit artifacts each time you restart your container,
it's possible to inform CUDA.jl which toolkit to use. This can be done by calling
`CUDA.set_runtime_version!` when building the container, after which a subsequent import
of CUDA.jl will download the necessary artifacts.

At run time you obviously do need a CUDA-compatible GPU as well as the CUDA driver library
to interface with it. Typically, that library is imported from the host system, e.g., by
launching `docker` using the `--gpus=all` flag:

```
$ docker run --rm -it --gpus=all julia

julia> using CUDA

julia> CUDA.versioninfo()
CUDA runtime 11.8
CUDA driver 11.8
NVIDIA driver 520.56.6

...
```

All of the above is demonstrated in the Dockerfile that's part of the CUDA.jl repository.

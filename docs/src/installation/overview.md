# [Overview](@id InstallationOverview)

The Julia CUDA stack requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and corresponding [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). The former should be installed by you
or your system administrator, while the latter can be automatically downloaded by Julia
using the artifact subsystem.



## Platform support

All three major operation systems are supported: Linux, Windows and macOS. However, that
support is subject to NVIDIA providing a CUDA toolkit for your system, subsequently macOS
support might be deprecated soon.

Similarly, we support x86, ARM, PPC, ... as long as Julia is supported on it and there
exists an NVIDIA driver and CUDA toolkit for your platform. The main development platform
(and the only CI system) however is x86_64 on Linux, so if you are using a more exotic
combination there might be bugs.



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
called `libcuda.dll` on Linux, `libcuda.dylib` on macOS and `nvcuda64.dll` on Windows. You
should also be able to execute the `nvidia-smi` command, which lists all available GPUs you
have access to.

Finally, to be able to use all of the Julia GPU stack you need to have permission to profile
GPU code. On Linux, that means loading the `nvidia` kernel module with the
`NVreg_RestrictProfilingToAdminUsers=0` option configured (e.g., in `/etc/modprobe.d`).
Refer to the [following
document](https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters)
for more information.



## CUDA toolkit

There are two different options to provide CUDA: either you [install it
yourself](https://developer.nvidia.com/cuda-downloads) in a way that is discoverable by the
Julia CUDA packages, or you let the packages download CUDA from artifacts. If you can use
artifacts (i.e., you are not using an unsupported platform or have no specific
requirements), it is recommended to do so: The CUDA toolkit is tightly coupled to the NVIDIA
driver, and compatibility is automatically taken into account when selecting an artifact to
use.


### Artifacts

Use of artifacts is the default option: Importing CUDA.jl will automatically download CUDA
upon first use of the API. You can inspect details about the process by enabling debug
logging:

```
$ JULIA_DEBUG=CUDA julia

julia> using CUDA

julia> CUDA.version()
┌ Debug: Trying to use artifacts...
└ @ CUDA CUDA/src/bindeps.jl:52
┌ Debug: Using CUDA 10.2.89 from an artifact at /home/tim/Julia/depot/artifacts/93956fcdec9ac5ea76289d25066f02c2f4ebe56e
└ @ CUDA CUDA/src/bindeps.jl:108
v"10.2.89"
```


### Local installation

If artifacts are unavailable for your platform, the Julia CUDA packages will look for a
local CUDA installation:

```
julia> CUDA.version()
┌ Debug: Trying to use artifacts...
└ @ CUDA CUDA/src/bindeps.jl:52
┌ Debug: Could not find a compatible artifact.
└ @ CUDA CUDA/src/bindeps.jl:73

┌ Debug: Trying to use local installation...
└ @ CUDA CUDA/src/bindeps.jl:114
┌ Debug: Found local CUDA 10.0.326 at /usr/local/cuda-10.0/targets/aarch64-linux, /usr/local/cuda-10.0
└ @ CUDA CUDA/src/bindeps.jl:141
v"10.0.326"
```

You might want to disallow use of artifacts, e.g., because an optimized CUDA installation is
available for your system. You can do so by setting the environment variable
`JULIA_CUDA_USE_BINARYBUILDER` to `false`.

To troubleshoot discovery of a local CUDA installation, you can set `JULIA_DEBUG=CCUDA` and
see the various paths where CUDA.jl looks. By setting any of the `CUDA_HOME`, `CUDA_ROOT` or
`CUDA_PATH` environment variables, you can guide the package to a specific directory.

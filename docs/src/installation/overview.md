# [Overview](@id InstallationOverview)

The Julia CUDA stack requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and matching [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). For now, both of these components
should be manually installed. If you are a Linux user, you should consider installing these
dependencies using a package manager instead of downloading them from the NVIDIA homepage;
refer to your distribution's documentation for more details.

To make sure you have everything set-up, you can try executing some of the applications that
the driver and toolkit provide. On Linux, you can verify driver availability by executing
`nvidia-smi`, and you have installed CUDA successfully if you can execute `ptxas --version`.


## CUDA discovery

Once you've installed the NVIDIA driver and CUDA toolkit, the Julia CUDA packages should
automatically pick up your installation by means of the functionality in CUDAapi.jl. Some
guidelines to make sure this works:

- CUDA driver: the driver library should be loadable with Libdl (e.g.,
  `Libdl.dlopen("libcuda")`)
- CUDA toolkit: the CUDA binaries should be on `PATH`

Alternatively, you can use the `CUDA_HOME` environment variable to point to an installation
of the CUDA toolkit.

To debug this, set `JULIA_DEBUG=CUDAapi` (or more generally `JULIA_DEBUG=all`) for details
on which paths are probed. If you file an issue, always include this information.


### Multiple CUDA toolkits

Generally, multiple installed CUDA toolkits are no supported because this may lead to
incompatible libraries being picked up. However, if you use the `CUDA_HOME` environment
variable to point to an installation, all other discovery heuristics will be disabled. This
should result in only that version of the CUDA toolkit being used, on the condition no other
toolkit is present in the global environment (`PATH`, `LD_LIBRARY_PATH`).


## Version compatibility

You should always use a CUDA toolkit that is supported by your driver. That means that the
toolkit version number should be lower or equal than the CUDA API version that is supported
by your driver (only taking into account the major and minor part of the version number).

Both these versions can be queried using the tools mentioned above, but you can also use the
Julia packages:

```julia
julia> using CUDAdrv, CUDAnative

julia> CUDAdrv.version() # CUDA toolkit supported by the driver
v"10.2.0"

julia> CUDAnative.version() # CUDA toolkit installed
v"10.2.89"
```

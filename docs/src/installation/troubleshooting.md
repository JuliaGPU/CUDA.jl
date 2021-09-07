# Troubleshooting


## Could not find a suitable CUDA installation

This means that CUDA.jl could not find or provide a CUDA toolkit. For more information,
re-run with the `JULIA_DEBUG` environment variable set to `CUDA`.

If you're encountering this error when disabling artifacts through by setting
`JULIA_CUDA_USE_BINARYBUILDER=false`, it is your own responsibility to make sure CUDA.jl
can detect the necessary pieces, e.g., by putting CUDA's binaries and libraries in
discoverable locations (i.e. on PATH, and on the library search path). Additionally, the
`CUDA_HOME` environment can be used to point CUDA.jl to where the CUDA toolkit is installed,
but that will only help if the contents of that directory have not been reorganized.


## UNKNOWN_ERROR(999)

If you encounter this error, there are several known issues that may be causing it:

- a mismatch between the CUDA driver and driver library: on Linux, look for clues in `dmesg`
- the CUDA driver is in a bad state: this can happen after resume. **Try rebooting**.

Generally though, it's impossible to say what's the reason for the error, but Julia is
likely not to blame. Make sure your set-up works (e.g., try executing `nvidia-smi`, a CUDA C
binary, etc), and if everything looks good file an issue.


## NVML library not found (on Windows)

Check and make sure the `NVSMI` folder is in your `PATH`. By default it may not be. Look in
`C:\Program Files\NVIDIA Corporation` for the `NVSMI` folder - you should see `nvml.dll`
within it. You can add this folder to your `PATH` and check that `nvidia-smi` runs properly.


## The specified module could not be found (on Windows)

Ensure the [Visual C++ Redistributable](https://aka.ms/vs/16/release/vc_redist.x64.exe) is
installed.

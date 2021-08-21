# Troubleshooting


## CUDA toolkit does not contain `XXX`

This means that you have an incomplete or missing CUDA toolkit, or that not all required
parts of the toolkit are discovered. Make sure the missing binary is present on your system,
and fix your CUDA toolkit installation if it isn't. Else, if you installed CUDA at a
nonstandard location, use the `CUDA_HOME` environment variable to direct Julia to that
location.

Note that this error only occurs when you're not using the automatic, artifact-based
installation (e.g., because you set `JULIA_CUDA_USE_BINARYBUILDER=false`). This is not a
recommended set-up, and it is possible that local CUDA discovery will be removed at some
point in the future.


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

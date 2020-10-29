# Troubleshooting


## CUDA toolkit does not contain `XXX`

This means that you have an incomplete or missing CUDA toolkit, or that not all required
parts of the toolkit are discovered. Make sure the missing binary is present on your system,
and fix your CUDA toolkit installation if it isn't. Else, if you installed CUDA at a
nonstandard location, use the `CUDA_HOME` environment variable to direct Julia to that
location.


## UNKNOWN_ERROR(999)

If you encounter this error, there are several known issues that may be causing it:

- a mismatch between the CUDA driver and driver library: on Linux, look for clues in `dmesg`
- the CUDA driver is in a bad state: this can happen after resume. Try rebooting.

Generally though, it's impossible to say what's the reason for the error, but Julia is
likely not to blame. Make sure your set-up works (e.g., try executing `nvidia-smi`, a CUDA C
binary, etc), and if everything looks good file an issue.


## NVML library not found (on Windows)

Check and make sure the `NVSMI` folder is in your `PATH`. By default it may not be. Look in
`C:\Program Files\NVIDIA Corporation` for the `NVSMI` folder - you should see `nvml.dll`
within it. You can add this folder to your `PATH` and check that `nvidia-smi` runs properly.


## LLVM error: Cannot cast between two non-generic address spaces

You are using an unpatched copy of LLVM, likely caused by using Julia as packaged by your
Linux distribution. These often decide to use a global copy of LLVM instead of using the one
built and patched by Julia during the compilation process. This is not supported: LLVM cannot
easily be used like a regular shared library, as Julia (and other users of LLVM alike) has an
extensive list of patches to be applied to the specific versions of LLVM that are supported.

It is thus recommended to use the official binaries, or use a version of Julia built without
setting `USE_SYSTEM_LLVM=1` (which you can suggest to maintainers of your Linux distribution).


## LoadError: UndefVarError: AddrSpacePtr not defined

You are using an old version of CUDA.jl in combination with a recent version of Julia
(1.5+). This is not supported, and you should be using CUDA.jl 1.x or above.

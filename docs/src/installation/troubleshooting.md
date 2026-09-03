# Troubleshooting


## Getting more details about CUDA errors

Many CUDA API errors are generic (e.g. `ERROR_INVALID_VALUE` or `ERROR_NOT_SUPPORTED`) and
do not explain what went wrong. With CUDA 12.9 or newer, the driver keeps an error log with
plain-English explanations of failed API calls, and CUDA.jl includes these messages when it
reports a `CuError`:

```
CUDA error: operation not supported (code 801, ERROR_NOT_SUPPORTED)
Driver log:
  [12:34:56.789][1234][CUDA][E] ...
  [12:34:56.789][1234][CUDA][E] Returning 801 (CUDA_ERROR_NOT_SUPPORTED) from cuModuleLoadDataEx
```

The same log can be written out by the driver directly, which is useful when an error is
swallowed by another library or when the process crashes: set the `CUDA_LOG_FILE`
environment variable to `stdout`, `stderr`, or a path before starting Julia.


## UndefVarError: libcuda not defined

This means that CUDA.jl could not find a suitable CUDA driver. For more information,
re-run with the `JULIA_DEBUG` environment variable set to `CUDA_Driver_jll`.


## UNKNOWN_ERROR(999)

If you encounter this error, there are several known issues that may be causing it:

- a mismatch between the CUDA driver and driver library: on Linux, look for clues in `dmesg`.
- some issue with the forwards-compatible driver library: try running with `JULIA_CUDA_USE_COMPAT=false` (or set the equivalent preference).
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

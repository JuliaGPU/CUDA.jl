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

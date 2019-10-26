# Overview

## Key packages

The following packages make up the Julia/CUDA stack:

* CUDAapi.jl: reusable CUDA components, e.g., discovery of libraries, compatibility of
  versions, etc.
* CUDAdrv.jl: low-level interface to the CUDA driver.
* CUDAnative.jl: native kernel programming, and interfaces to related tools and libraries.
* CuArrays.jl: high-level array programming interface, and integration with NVIDIA vendor
  libraries.

Most users should **start out with CuArrays.jl**, as the package provides an easy way to use
GPUs with optimized implementations of common array operations. This includes higher-order
operations such as map and broadcast, which provide great flexibility to perform custom
operations on the GPU.

If you run into operations that not supported by CuArrays.jl, or your application requires
verify specific functionality that cannot easily be expressed with other array operations,
you might need to **use CUDAnative.jl to implement custom kernels**. This package is the
counterpart of CUDA C kernel programming, and although the use of the Julia programming
language makes it much more productive, it is still a low-level programming interface that
requires knowledge about GPU programming.

Similarly, you might need **CUDAdrv.jl to call specific CUDA API functions** that are not
yet covered by the high-level array programming interface from CuArrays.jl. For example, if
you need to launch operations on a stream, or want to program multiple GPUs from a single
process, you will need to interface with CUDAdrv.jl directly. Over time, this operations
might find their way into CuArrays.jl in the form of a high-level interface that doesn't
require experience with the CUDA APIs.

Finally, if your package only uses GPUs conditionally, you should use **CUDAapi.jl to
determine GPU availability**. This package is the only one that is guaranteed to
successfully import; other packages will throw an error if you use them without CUDA or an
available GPU.

# Overview

The CUDA.jl package depends on and reexports functionality from several other packages:

* CUDAapi.jl: reusable CUDA components, e.g., discovery of libraries, compatibility of
  versions, etc.
* CUDAdrv.jl: low-level interface to the CUDA driver.
* CUDAnative.jl: native kernel programming, and interfaces to related tools and libraries.
* CuArrays.jl: high-level array programming interface, and integration with NVIDIA vendor
  libraries.

The following sections document the use of functionality from each of these packages. Don't
spend time reading all of it before you make sure you actually need to: much of the Julia
CUDA programming stack can be used by just relying on the `CuArray` type, and otherwise
using platform-agnostic programming patterns like `broadcast` and other array abstractions.

A typical workflow looks as follows:

1. develop an application using generic array functionality, and test it on the CPU with the
   `Array` type
2. port your application to the GPU by switching to the `CuArray` type
3. disallow the CPU fallback ("scalar indexing") to find operations that are not implemented
   for or incompatible with GPU execution
4. (optional) use lower-level, CUDA-specific interfaces to implement missing functionality
   or optimize performance

# Major

* Improve installability: no patched Julia, out-of-the-box experience

* Function dispatch on device (hardware support, warp size, ...). Use this to safely reuse
  and override parts of Base.



# Minor

* Use LLVM's linker with `InternalizeLinkedSymbols` when linking libdevice
  (simplifies code in jit.jl)

* Pass 'Val{T}()' instead of 'Val{T}', easier on the compiler


## CUDA support/interfacing

* ReadOnlyArray --> `getindex` does `__ldg`


# Ideas

* Make `CuArray` immutable such that we can merge it with `CuDeviceArray` (needs to be
  immutable to allocate within device code, eg. shared memory).

* Specialization on kernel size (possibly only when specifying constant dimensions): replace
  calls to dimension intrinsics with constants, and avoid branches (eg. bounds checks) by
  combining constant dimensions with proper `!range` metadata on index intrinsics.

* `CuStaticArray` with size as compile-time parameter, as base for static shared memory
  (while dynamic shared memory uses regular `CuDeviceArray`s)

* Try using NVIDIA's NVVM instead of LLVM's PTX back-end. This might give us DWARF, as well
  as improved performance. But it involves generating NVVM compatible IR, possibly
  duplicating a lot of code in the front-end (eg. intrinsics, libdevice, etc) and requiring
  us to target an older LLVM IR format (see [Android's forward-ported
  BitWriter](https://android.googlesource.com/platform/frameworks/compile/slang/+/master)).

# Major

* Improve installability: no patched Julia, out-of-the-box experience

* Function dispatch on device (hardware support, warp size, ...). Use this to safely reuse
  and override parts of Base.

* Reduce launch cost: no synchronization on kernel parameters malloc/memcpy


# Minor

* LLVM 4.0: NVVM reflect params have changed, now only accepted via module parameter (see D28700).


## CUDA support/interfacing

* ReadOnlyArray --> `getindex` does `__ldg`

* Proper address space support, for different memories and function arguments. On the other
  hand, is this necessary? Just mark the AS, convert as soon as possible, and use the
  inference pass to improve performance.



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

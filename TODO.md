# Major

* Improve installability: no patched Julia, out-of-the-box experience

* Function dispatch on device (hardware support, warp size, ...). Use this to safely reuse
  and override parts of Base.

* Reduce launch cost: no synchronization on kernel parameters malloc/memcpy



# Minor

* Disable bounds checking if `cuda-memcheck` is detected (`CUDA_MEMCHECK` env var), and emit
  line-number information if any debug tool is detected

* Rename `code_native` to `code_ptx`, and provide `code_sass`


## CUDA support/interfacing

* Benchmarking: surround kernel invocation with `@benchmark` --> insert perf events, etc

* ReadOnlyArray --> `getindex` does `__ldg`

* Proper address space support, for different memories and function arguments. On the other
  hand, is this necessary? Just mark the AS, convert as soon as possible, and use the
  inference pass to improve performance.


## Julia support

* Step ranges



# Ideas

* Make `CuArray` immutable such that we can merge it with `CuDeviceArray` (needs to be
  immutable to allocate within device code, eg. shared memory).

* Specialization on kernel size (possibly only when specifying constant dimensions): replace
  calls to dimension intrinsics with constants, and avoid branches (eg. bounds checks) by
  combining constant dimensions with proper `!range` metadata on index intrinsics.

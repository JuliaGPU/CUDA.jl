# Major

* Integrate target-selection with dispatch, including a "generic" target. Use this eg. to
  expose math functions in a sane manner, and avoid the JIT clash of running the same
  functions on device and host side.

* Better way of integrating with call semantics (eg. ghost types not being passed)

* Proper address space support, for different memories and function arguments. On the other
  hand, is this necessary? Just mark the AS, convert as soon as possible, and use the
  inference pass to improve performance.


# Minor

* Make `CuArray` immutable such that we can merge it with `CuDeviceArray` (needs to be
  immutable to allocate within device code, eg. shared memory).

* Support for non-escaping boxed objects (tough, see JuliaLang/Julia#12205)

* Global variables? Closure variables?

* DWARF debugging, currently not supported by the back-end (`.debug_info` missing):
  ```c
  const char *const argv_nvptxdbg[] = {"", "-debug-compile"};
  cl::ParseCommandLineOptions(sizeof(argv_nvptxdbg)/sizeof(argv_nvptxdbg[0]), argv_nvptxdbg, "nvptx-debug-compile\n");
  ```

* Disable bounds checking if `cuda-memcheck` is used (`CUDA_MEMCHECK` env var)

* Benchmarking: surround kernel invocation with `@benchmark` --> insert perf events, etc


# Optimizations

* Specialization on kernel size (possibly only when specifying constant dimensions): replace
  calls to dimension intrinsics with constants, and avoid branches (eg. bounds checks) by
  combining constant dimensions with proper `!range` metadata on index intrinsics.

* ReadOnlyArray --> `getindex` does `__ldg`

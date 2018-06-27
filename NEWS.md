CUDAnative v1.0 release notes
=============================

This document describes major features and user-facing changes to CUDAnative.


New features
------------

  * `@device_code_...` macros make it easy to inspect generated device code even
    if the outermost function call isn't a `@cuda` invocation. This is especially
    useful in combination with, e.g., CuArrays. The `@device_code` macro dumps
    _all_ forms of intermediate code to a directory, for easy inspection ([#147]).

  * Fast versions of CUDA math intrinsics are now wrapped ([#152]).

  * Support for loading values through the texture cache, aka. `__ldg`, has been
    added. No `getindex`-based interfaced is available yet, manually use
    `unsafe_cached_load` instead ([#158]).

  * Multiple devices are supported, by calling `device!` to switch to another
    device. The CUDA API is now also initialized lazily, so be sure to call
    `device!` before performing any work to avoid allocating a context on device
    0 ([#175]).

  * Support for object and closure kernel functions has been added ([#176]).

  * The return value of kernels is now automatically ignored, so no need for
    `return [nothing]` anymore ([#178]).


Changes
-------

  * Debug info generation now honors the `-g` flag as passed to the Julia command,
    and is no longer tied to the `DEBUG` environment variable.

  * Log messages are implemented using the new Base Julia logging system. Debug
    logging can be enabled by specifying the `JULIA_DEBUG=CUDAnative` environment
    variable.

  * All functions not marked `@noinline` are forcibly inlined into the
    entry-point. Although coarse, this often improves performance ([#151]).

  * The syntax of `@cuda` now takes keyword arguments, eg. `@cuda threads=1
    foo(...)`, instead of the old tuple syntax. See the documentation of `@cuda`
    for a list of supported arguments ([#154]).

  * Non isbits values can be passed to a kernel, as long as they are unused. This
    makes it easier to implement GPU-versions of existing functions, without
    requiring a different method signature ([#168]).

  * Indexing intrinsics now return `Int`, so no need to convert to `(U)Int32`
    anymore. Although this might require more registers, it allows LLVM to
    simplify code ([#182]).

  * Better error messages, showing backtraces into GPU code (#189) and detecting
    common pitfalls like recursion or use of Base intrinsics (#210).


Deprecations
------------

  * `CUDAnative.@profile` has been removed, use `CUDAdrv.@profile` with a manual
    warm-up step instead.

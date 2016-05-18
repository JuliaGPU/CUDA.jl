# Code generation

* Make `sm_35` (and corresponding `libdevice`) configurable

* Add a `--debug` option, generating `.loc` instructions, exposed to
  `CUDAnative.jl` in order to properly configure the PTX JIT.

* `@target` should allow for generic functions (ie. for all targets).

* `@target` should be incorporated in dispatch, selecting functions from the
  same or a generic target. This is especially useful for device intrinsics (eg.
  `sin`, `cos`, ...); as a workaround, these aren't exported currently.

* Use the existing `cfunction` functionality to generate a CUDA kernels?

* See [ISPC.jl](https://github.com/damiendr/ISPC.jl) for extracting closure vars
  (globals and such)


# API wrapping

* Fix the shared memory: wrap in a type, don't refer to the global through
  `llvmcall` and add support for statically defined shared memory.


# Bugs

* Run with `--inline=no`

* Running with `--code-coverage=user` doesn't work with `TRACE=1` because PTX
  methods get partially registered in the host module due to the `code_*` calls
  (I think)


# Performance

* Additional specialization: when invoking a kernel with constant dimension config,
  remove the intrinsic lookups and specialize for that configuration.

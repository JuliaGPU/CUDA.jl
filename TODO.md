# Package infrastructure

* Intrinsics clobber: importing CUDA overrides the normal stuff (ie. `floor` or
  `sin`)... Current work-around: don't export, require `CUDA.`. Make them
  conditional, based on `@target ptx`?

* Use the existing `cfunction` functionality to generate a CUDA kernels?


# Native execution

* Fix the shared memory: wrap in a type, don't refer to the global through
  `llvmcall` and add support for statically defined shared memory.

* Pass the native codegen context into `@cuda`, allowing for multiple active
  contexts.

* Related to the point above, ff we have a single `main`, instantiating `cgctx`
  and launching a kernel (a common use-case, right?) the macro expansion happens
  before the `cgctx` instantiation is executed, but the compiler _does_ use the
  codegen context internally!

* Truly prohibit boxes (currently only a warning). This requires reworking
  exception support, as we currently still create the exception, but ignore it
  in `jl_throw` causing the box to disappear later on.

* See [ISPC.jl](https://github.com/damiendr/ISPC.jl) for extracting closure vars
  (globals and such)

* Run with `--inline=no`

* Running with `--code-coverage=user` doesn't work with `TRACE=1` because PTX
  methods get partially registered in the host module due to the `code_*` calls
  (I think)

* Additional specialization: when invoking a kernel with constant dimension config,
  remove the intrinsic lookups and specialize for that configuration.

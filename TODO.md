# Major

* Provide a third codegen target, for generic code

* Disconnect or disambiguate the compiler's state based on the codegen target
  (`functionObject`, etc). This will allow compiling the same functions for
  multiple targets (see `bugs/host_after_ptx.jl`)

* Dispatch based on the callee's `@target`, and expose math functions with it

* Properly integrate kernel calling, as actual arguments might not correspond
  with source-level arguments (see `bugs/inlined_parameters.jl`)

* Replace generated function-time code generation (and consequent calls into
  type inference) *without* degrading performance of calling kernels (ie. avoid
  runtime code). Use existing `cfunction` functionality? See
  JuliaLang/julia#15942, JuliaLang/julia#16000.

* Fix the shared memory: wrap in a type, don't refer to the global through
  `llvmcall` and add support for statically defined shared memory

* Related to shared memory: proper address space support, for different memories
  and function arguments


# Minor

* Support for non-escaping boxed objects (tough, see JuliaLang/Julia##12205)

* Printing from kernels using `vprintf` (needs unboxed string literals)

* Global variables? Closure variables?

* Make `sm_35` (and corresponding `libdevice` filename) configurable

* Add a `--debug` option, generating `.loc` instructions, exposed to
  `CUDAnative.jl` in order to properly configure the PTX JIT

* Fix running tests with `--inline=no`

* A lot of undefined references running valgrind -- check this out!

* Allow "@target ptx \ \n function"

* Reorganize test testsets, they are bad

* Verify the dimension calculations in the tests and examples (cfr. blackscholes slowdown)


# Ideas

* Additional specialization: when invoking a kernel with constant dimension
  config, remove the intrinsic lookups and specialize for that configuration.

# Major

* Provide a third codegen target, for generic code

* Disconnect or disambiguate the compiler's state based on the codegen target
  (`functionObject`, etc). This will allow compiling the same functions for
  multiple targets (see `bugs/host_after_ptx.jl`)

* Dispatch based on the callee's `@target`, and expose math functions with it

* Better way of integrating with call semantics (eg. ghost types not being passed)

* Proper address space support, for different memories and function arguments. On the other
  hand, is this necessary? Just mark the AS, convert as soon as possible, and use the
  inference pass to improve performance.

* Make `CuDeviceArray` a proper (immutable) type: this requires being able to pass the
  entire object on the stack, or we would need copies to and from GPU memory before _every_
  kernel launch. It would allow proper bounds checking, eg. for static shared memory.


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

* Verify the dimension calculations in the tests and examples (cfr. blackscholes slowdown)


# Ideas

* Additional specialization: when invoking a kernel with constant dimension
  config, remove the intrinsic lookups and specialize for that configuration.

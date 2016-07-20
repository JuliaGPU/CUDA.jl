# Major

* Integrate target-selection with dispatch, including a "generic" target. Use this eg. to
  expose math functions in a sane manner, and avoid the JIT clash of running the same
  functions on device and host side.

* Better way of integrating with call semantics (eg. ghost types not being passed)

* Proper address space support, for different memories and function arguments. On the other
  hand, is this necessary? Just mark the AS, convert as soon as possible, and use the
  inference pass to improve performance.

* Stream granularity for execution, exceptions, etc


# Minor

* Make `CuArray` immutable such that we can merge it with `CuDeviceArray` (needs to be
  immutable to allocate within device code, eg. shared memory).

* Support for non-escaping boxed objects (tough, see JuliaLang/Julia#12205)

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

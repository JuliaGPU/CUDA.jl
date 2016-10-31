* The old `warpsize` integer constant is now a function, and is not a compile-time constant
  anymore. Instead, it returns the contents of a PTX-specific variable, and should thus only
  be used in device code. For a host-side/compile-time value, use CUDAdrv's
  `warpsize(dev::CuDevice)` instead.
* We now provide `unsafe_convert` to convert CUDA objects (`CuModule`, `CuDevice`, ...) to pointers.
  Shouldn't we use `Base.cconvert`?

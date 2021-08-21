# Essentials


## Initialization

```@docs
CUDA.functional(::Bool)
has_cuda
has_cuda_gpu
```


## Global state

```@docs
context
context!(::CuContext)
context!(::Function, ::CuContext)
device
device!(::CuDevice)
device!(::Function, ::CuDevice)
device_reset!
stream
stream!(::CuStream)
stream!(::Function, ::CuStream)
```

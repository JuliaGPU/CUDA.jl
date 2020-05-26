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
device!(::CuDevice)
device!(::Function, ::CuDevice)
device_reset!
```

If you have a library or application that maintains its own global state, you might need to
react to context or task switches:

```@docs
CUDA.attaskswitch
CUDA.atcontextswitch
```

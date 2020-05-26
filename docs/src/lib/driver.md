# CUDA driver

This section lists the package's public functionality that directly corresponds to
functionality of the CUDA driver API. In general, the abstractions stay close to those of
the CUDA driver API, so for more information on certain library calls you can consult the
[CUDA driver API reference](http://docs.nvidia.com/cuda/cuda-driver-api/).

The documentation is grouped according to the modules of the driver API.


## Error Handling

```@docs
CuError
name(::CuError)
CUDA.description(::CuError)
```


## Version Management

```@docs
CUDA.version()
```


## Device Management

```@docs
CuDevice
devices
name(::CuDevice)
totalmem(::CuDevice)
attribute
```

Certain common attributes are exposed by additional convenience functions:

```@docs
capability(::CuDevice)
warpsize(::CuDevice)
```


## Context Management

```@docs
CuContext
CUDA.unsafe_destroy!(::CuContext)
CuCurrentContext
activate(::CuContext)
synchronize()
device(::CuContext)
```

### Primary Context Management

```@docs
CuPrimaryContext
CuContext(::CuPrimaryContext)
isactive(::CuPrimaryContext)
flags(::CuPrimaryContext)
setflags!(::CuPrimaryContext, ::CUDA.CUctx_flags)
unsafe_reset!(::CuPrimaryContext)
CUDA.unsafe_release!(::CuContext)
```


## Module Management

```@docs
CuModule
```

### Function Management

```@docs
CuFunction
```

### Global Variable Management

```@docs
CuGlobal
eltype(::CuGlobal)
Base.getindex(::CuGlobal)
Base.setindex!(::CuGlobal{T}, ::T) where {T}
```

### Linker

```@docs
CuLink
add_data!
add_file!
CuLinkImage
complete
CuModule(::CuLinkImage, args...)
```


## Memory Management

Three kinds of memory buffers can be allocated: device memory, host memory, and unified
memory. Each of these buffers can be allocated by calling `alloc` with the type of buffer as
first argument, and freed by calling `free`. Certain buffers have specific methods defined.

```@docs
Mem.DeviceBuffer
Mem.alloc(::Type{Mem.DeviceBuffer}, ::Integer)
```

```@docs
Mem.HostBuffer
Mem.alloc(::Type{Mem.HostBuffer}, ::Integer, flags)
Mem.register(::Type{Mem.HostBuffer}, ::Ptr, ::Integer, flags)
Mem.unregister(::Mem.HostBuffer)
```

```@docs
Mem.UnifiedBuffer
Mem.alloc(::Type{Mem.UnifiedBuffer}, ::Integer, ::CUDA.CUmemAttach_flags)
Mem.prefetch(::Mem.UnifiedBuffer, bytes::Integer; device, stream)
Mem.advise(::Mem.UnifiedBuffer, ::CUDA.CUmem_advise, ::Integer; device)
```

To work with these buffers, you need to `convert` them to a `Ptr` or `CuPtr`. Several
methods then work with these raw pointers:



### Memory info

```@docs
CUDA.available_memory
CUDA.total_memory
```


## Stream Management

```@docs
CuStream
CuDefaultStream
synchronize(::CuStream)
```

## Event Management

```@docs
CuEvent
record
synchronize(::CuEvent)
elapsed
CUDA.@elapsed
```

## Execution Control

```@docs
CuDim3
cudacall
CUDA.launch
```

## Profiler Control

```@docs
CUDA.@profile
CUDA.Profile.start
CUDA.Profile.stop
```

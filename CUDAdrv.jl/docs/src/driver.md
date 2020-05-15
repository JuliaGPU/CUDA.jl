# CUDA driver

This section lists the package's public functionality that directly corresponds to
functionality of the CUDA driver API. In general, the abstractions stay close to those of
the CUDA driver API, so for more information on certain library calls you can consult the
[CUDA driver API reference](http://docs.nvidia.com/cuda/cuda-driver-api/).

The documentation is grouped according to the modules of the driver API.


## Error Handling

```@docs
CUDAdrv.CuError
CUDAdrv.name(::CuError)
CUDAdrv.description(::CuError)
```


## Version Management

```@docs
CUDAdrv.version()
```


## Device Management

```@docs
CUDAdrv.CuDevice
CUDAdrv.devices
CUDAdrv.name(::CuDevice)
CUDAdrv.totalmem(::CuDevice)
CUDAdrv.attribute
```

Certain common attributes are exposed by additional convenience functions:

```@docs
CUDAdrv.capability(::CuDevice)
CUDAdrv.warpsize(::CuDevice)
```


## Context Management

```@docs
CUDAdrv.CuContext
CUDAdrv.destroy!(::CuContext)
CUDAdrv.CuCurrentContext
CUDAdrv.activate(::CuContext)
CUDAdrv.synchronize()
CUDAdrv.device(::CuContext)
```

### Primary Context Management

```@docs
CUDAdrv.CuPrimaryContext
CUDAdrv.CuContext(::CuPrimaryContext)
CUDAdrv.isactive(::CuPrimaryContext)
CUDAdrv.flags(::CuPrimaryContext)
CUDAdrv.setflags!(::CuPrimaryContext, ::CUDAdrv.CUctx_flags)
CUDAdrv.unsafe_reset!(::CuPrimaryContext, ::Bool)
```


## Module Management

```@docs
CUDAdrv.CuModule
```

### Function Management

```@docs
CUDAdrv.CuFunction
```

### Global Variable Management

```@docs
CUDAdrv.CuGlobal
CUDAdrv.eltype(::CuGlobal)
Base.getindex(::CuGlobal)
Base.setindex!(::CuGlobal{T}, ::T) where {T}
```

### Linker

```@docs
CUDAdrv.CuLink
CUDAdrv.add_data!
CUDAdrv.add_file!
CUDAdrv.CuLinkImage
CUDAdrv.complete
CUDAdrv.CuModule(::CUDAdrv.CuLinkImage, args...)
```


## Memory Management

Three kinds of memory buffers can be allocated: device memory, host memory, and unified
memory. Each of these buffers can be allocated by calling `alloc` with the type of buffer as
first argument, and freed by calling `free`. Certain buffers have specific methods defined.

```@docs
CUDAdrv.Mem.DeviceBuffer
CUDAdrv.Mem.alloc(::Type{Mem.DeviceBuffer}, ::Integer)
```

```@docs
CUDAdrv.Mem.HostBuffer
CUDAdrv.Mem.alloc(::Type{Mem.HostBuffer}, ::Integer, flags)
CUDAdrv.Mem.register(::Type{Mem.HostBuffer}, ::Ptr, ::Integer, flags)
CUDAdrv.Mem.unregister(::Mem.HostBuffer)
```

```@docs
CUDAdrv.Mem.UnifiedBuffer
CUDAdrv.Mem.alloc(::Type{Mem.UnifiedBuffer}, ::Integer, ::CUDAdrv.CUmemAttach_flags)
CUDAdrv.Mem.prefetch(::Mem.UnifiedBuffer, bytes::Integer; device, stream)
CUDAdrv.Mem.advise(::Mem.UnifiedBuffer, ::CUDAdrv.CUmem_advise, ::Integer; device)
```

To work with these buffers, you need to `convert` them to a `Ptr` or `CuPtr`. Several
methods then work with these raw pointers:



### Memory info

```@docs
CUDAdrv.available_memory
CUDAdrv.total_memory
```


## Stream Management

```@docs
CUDAdrv.CuStream
CUDAdrv.CuDefaultStream
CUDAdrv.synchronize(::CuStream)
```

## Event Management

```@docs
CUDAdrv.CuEvent
CUDAdrv.record
CUDAdrv.synchronize(::CuEvent)
CUDAdrv.elapsed
CUDAdrv.@elapsed
```

## Execution Control

```@docs
CUDAdrv.CuDim3
CUDAdrv.cudacall
CUDAdrv.launch
```

## Profiler Control

```@docs
CUDAdrv.@profile
CUDAdrv.Profile.start
CUDAdrv.Profile.stop
```

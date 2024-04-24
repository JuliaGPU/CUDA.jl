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
CUDA.driver_version()
CUDA.system_driver_version()
CUDA.runtime_version()
CUDA.set_runtime_version!
CUDA.reset_runtime_version!
```


## Device Management

```@docs
CuDevice
devices
current_device
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
current_context
activate(::CuContext)
synchronize(::CuContext)
device_synchronize
```

### Primary Context Management

```@docs
CuPrimaryContext
CuContext(::CuPrimaryContext)
isactive(::CuPrimaryContext)
flags(::CuPrimaryContext)
setflags!(::CuPrimaryContext, ::CUDA.CUctx_flags)
unsafe_reset!(::CuPrimaryContext)
CUDA.unsafe_release!(::CuPrimaryContext)
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

Different kinds of memory objects can be created, representing different kinds of memory
that the CUDA toolkit supports. Each of these memory objects can be allocated by calling
`alloc` with the type of memory as first argument, and freed by calling `free`. Certain
kinds of memory have specific methods defined.

### Device memory

This memory is accessible only by the GPU, and is the most common kind of memory used in
CUDA programming.

```@docs
CUDA.DeviceMemory
CUDA.alloc(::Type{CUDA.DeviceMemory}, ::Integer)
```

### Unified memory

Unified memory is accessible by both the CPU and the GPU, and is managed by the CUDA
runtime. It is automatically migrated between the CPU and the GPU as needed, which
simplifies programming but can lead to performance issues if not used carefully.

```@docs
CUDA.UnifiedMemory
CUDA.alloc(::Type{CUDA.UnifiedMemory}, ::Integer, ::CUDA.CUmemAttach_flags)
CUDA.prefetch(::CUDA.UnifiedMemory, bytes::Integer; device, stream)
CUDA.advise(::CUDA.UnifiedMemory, ::CUDA.CUmem_advise, ::Integer; device)
```

### Host memory

Host memory resides on the CPU, but is accessible by the GPU via the PCI bus. This is the
slowest kind of memory, but is useful for communicating between running kernels and the
host (e.g., to update counters or flags).

```@docs
CUDA.HostMemory
CUDA.alloc(::Type{CUDA.HostMemory}, ::Integer, flags)
CUDA.register(::Type{CUDA.HostMemory}, ::Ptr, ::Integer, flags)
CUDA.unregister(::CUDA.HostMemory)
```

### Array memory

Array memory is a special kind of memory that is optimized for 2D and 3D access patterns. The memory is opaquely managed by the CUDA runtime, and is typically only used on combination with texture intrinsics.

```@docs
CUDA.ArrayMemory
CUDA.alloc(::Type{CUDA.ArrayMemory{T}}, ::Dims) where T
```

### Pointers

To work with these buffers, you need to `convert` them to a `Ptr`, `CuPtr`, or in the case
of `ArrayMemory` an `CuArrayPtr`. You can then use common Julia methods on these pointers,
such as `unsafe_copyto!`. CUDA.jl also provides some specialized functionality that does not
match standard Julia functionality:

```@docs
CUDA.unsafe_copy2d!
CUDA.unsafe_copy3d!
CUDA.memset
```

### Other

```@docs
CUDA.free_memory
CUDA.total_memory
```


## Stream Management

```@docs
CuStream
CUDA.isdone(::CuStream)
priority_range
priority
synchronize(::CuStream)
CUDA.@sync
```

For specific use cases, special streams are available:

```@docs
default_stream
legacy_stream
per_thread_stream
```

## Event Management

```@docs
CuEvent
record
synchronize(::CuEvent)
CUDA.isdone(::CuEvent)
CUDA.wait(::CuEvent)
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

## Texture Memory

Textures are represented by objects of type `CuTexture` which are bound to some underlying
memory, either `CuArray`s or `CuTextureArray`s:

```@docs
CUDA.CuTexture
CUDA.CuTexture(array)
```

You can create `CuTextureArray` objects from both host and device memory:

```@docs
CUDA.CuTextureArray
CUDA.CuTextureArray(array)
```

## Occupancy API

The occupancy API can be used to figure out an appropriate launch configuration for a
compiled kernel (represented as a `CuFunction`) on the current device:

```@docs
launch_configuration
active_blocks
occupancy
```

## Graph Execution

CUDA graphs can be easily recorded and executed using the high-level `@captured` macro:

```@docs
CUDA.@captured
```

Low-level operations are available too:

```@docs
CuGraph
capture
instantiate
launch(::CUDA.CuGraphExec)
update
```

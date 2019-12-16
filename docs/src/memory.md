# Memory management

Core interface:

```@docs
CuArrays.alloc
CuArrays.free
OutOfGPUMemoryError
```

Functionality to work with external allocations:

```@docs
CuArrays.reclaim
CuArrays.extalloc
```


## Utilities

```@docs
CuArrays.@allocated
CuArrays.@time
CuArrays.memory_status
```


## Debug timers

```@docs
CuArrays.enable_timings()
CuArrays.alloc_timings()
CuArrays.pool_timings()
CuArrays.reset_timers!()
```


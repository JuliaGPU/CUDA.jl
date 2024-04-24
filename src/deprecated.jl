using Base: @deprecate_binding

## removal of Mem submodule

export Mem
module Mem

using ..CUDA
using Base: @deprecate_binding

@deprecate_binding AbstractBuffer CUDA.AbstractMemory false
@deprecate_binding DeviceBuffer CUDA.DeviceMemory false
@deprecate_binding HostBuffer CUDA.HostMemory false
@deprecate_binding UnifiedBuffer CUDA.UnifiedMemory false
@deprecate_binding ArrayBuffer CUDA.ArrayMemory false

@deprecate_binding Device CUDA.DeviceMemory false
@deprecate_binding Host CUDA.HostMemory false
@deprecate_binding Unified CUDA.UnifiedMemory false

@deprecate alloc(args...) CUDA.alloc(args...) false
@deprecate free(args...) CUDA.free(args...) false

@deprecate_binding HOSTALLOC_PORTABLE CUDA.MEMHOSTALLOC_PORTABLE false
@deprecate_binding HOSTALLOC_DEVICEMAP CUDA.MEMHOSTALLOC_DEVICEMAP false
@deprecate_binding HOSTALLOC_WRITECOMBINED CUDA.MEMHOSTALLOC_WRITECOMBINED false

@deprecate_binding HOSTREGISTER_PORTABLE CUDA.MEMHOSTREGISTER_PORTABLE false
@deprecate_binding HOSTREGISTER_DEVICEMAP CUDA.MEMHOSTREGISTER_DEVICEMAP false
@deprecate_binding HOSTREGISTER_IOMEMORY CUDA.MEMHOSTREGISTER_IOMEMORY false

@deprecate register(args...) CUDA.register(args...) false
@deprecate unregister(args...) CUDA.register(args...) false

@deprecate_binding ATTACH_GLOBAL CUDA.MEM_ATTACH_GLOBAL false
@deprecate_binding ATTACH_HOST CUDA.MEM_ATTACH_HOST false
@deprecate_binding ATTACH_SINGLE CUDA.MEM_ATTACH_SINGLE false

@deprecate prefetch(args...) CUDA.prefetch(args...) false
@deprecate advise(args...) CUDA.prefetch(args...) false

@deprecate_binding ADVISE_SET_READ_MOSTLY CUDA.MEM_ADVISE_SET_READ_MOSTLY false
@deprecate_binding ADVISE_UNSET_READ_MOSTLY CUDA.MEM_ADVISE_UNSET_READ_MOSTLY false
@deprecate_binding ADVISE_SET_PREFERRED_LOCATION CUDA.MEM_ADVISE_SET_PREFERRED_LOCATION false
@deprecate_binding ADVISE_UNSET_PREFERRED_LOCATION CUDA.MEM_ADVISE_UNSET_PREFERRED_LOCATION false
@deprecate_binding ADVISE_SET_ACCESSED_BY CUDA.MEM_ADVISE_SET_ACCESSED_BY false
@deprecate_binding ADVISE_UNSET_ACCESSED_BY CUDA.MEM_ADVISE_UNSET_ACCESSED_BY false

@deprecate set!(args...) CUDA.memset(args...) false
@deprecate unsafe_copy2d!(args...) CUDA.unsafe_copy2d!(args...) false
@deprecate unsafe_copy3d!(args...) CUDA.unsafe_copy3d!(args...) false
@deprecate pin(args...) CUDA.pin(args...) false
@deprecate info() CUDA.memory_info() false

end

@deprecate memory_status() CUDA.pool_status() false

@deprecate available_memory() free_memory() false

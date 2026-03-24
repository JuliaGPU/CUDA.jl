using Base: @deprecate_binding

## removal of Mem submodule

export Mem
module Mem

using ..CUDACore
using Base: @deprecate_binding

@deprecate_binding AbstractBuffer CUDACore.AbstractMemory false
@deprecate_binding DeviceBuffer CUDACore.DeviceMemory false
@deprecate_binding HostBuffer CUDACore.HostMemory false
@deprecate_binding UnifiedBuffer CUDACore.UnifiedMemory false
@deprecate_binding ArrayBuffer CUDACore.ArrayMemory false

@deprecate_binding Device CUDACore.DeviceMemory false
@deprecate_binding Host CUDACore.HostMemory false
@deprecate_binding Unified CUDACore.UnifiedMemory false

@deprecate alloc(args...) CUDACore.alloc(args...) false
@deprecate free(args...) CUDACore.free(args...) false

@deprecate_binding HOSTALLOC_PORTABLE CUDACore.MEMHOSTALLOC_PORTABLE false
@deprecate_binding HOSTALLOC_DEVICEMAP CUDACore.MEMHOSTALLOC_DEVICEMAP false
@deprecate_binding HOSTALLOC_WRITECOMBINED CUDACore.MEMHOSTALLOC_WRITECOMBINED false

@deprecate_binding HOSTREGISTER_PORTABLE CUDACore.MEMHOSTREGISTER_PORTABLE false
@deprecate_binding HOSTREGISTER_DEVICEMAP CUDACore.MEMHOSTREGISTER_DEVICEMAP false
@deprecate_binding HOSTREGISTER_IOMEMORY CUDACore.MEMHOSTREGISTER_IOMEMORY false

@deprecate register(args...) CUDACore.register(args...) false
@deprecate unregister(args...) CUDACore.unregister(args...) false

@deprecate_binding ATTACH_GLOBAL CUDACore.MEM_ATTACH_GLOBAL false
@deprecate_binding ATTACH_HOST CUDACore.MEM_ATTACH_HOST false
@deprecate_binding ATTACH_SINGLE CUDACore.MEM_ATTACH_SINGLE false

@deprecate prefetch(args...) CUDACore.prefetch(args...) false
@deprecate advise(args...) CUDACore.prefetch(args...) false

@deprecate_binding ADVISE_SET_READ_MOSTLY CUDACore.MEM_ADVISE_SET_READ_MOSTLY false
@deprecate_binding ADVISE_UNSET_READ_MOSTLY CUDACore.MEM_ADVISE_UNSET_READ_MOSTLY false
@deprecate_binding ADVISE_SET_PREFERRED_LOCATION CUDACore.MEM_ADVISE_SET_PREFERRED_LOCATION false
@deprecate_binding ADVISE_UNSET_PREFERRED_LOCATION CUDACore.MEM_ADVISE_UNSET_PREFERRED_LOCATION false
@deprecate_binding ADVISE_SET_ACCESSED_BY CUDACore.MEM_ADVISE_SET_ACCESSED_BY false
@deprecate_binding ADVISE_UNSET_ACCESSED_BY CUDACore.MEM_ADVISE_UNSET_ACCESSED_BY false

@deprecate set!(args...) CUDACore.memset(args...) false
@deprecate unsafe_copy2d!(args...) CUDACore.unsafe_copy2d!(args...) false
@deprecate unsafe_copy3d!(args...) CUDACore.unsafe_copy3d!(args...) false
@deprecate pin(args...) CUDACore.pin(args...) false
@deprecate info() CUDACore.memory_info() false

end

@deprecate memory_status() CUDACore.pool_status() false

@deprecate available_memory() free_memory() false

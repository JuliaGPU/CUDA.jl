## compute-sanitizer annotations
#
# compute-sanitizer tracks the allocations made with the CUDA APIs, so it can't tell the
# blocks that the caching allocator carves out of a segment apart: accesses past the end of
# a block, or to a block that has been freed, go unnoticed as long as they stay within the
# segment. NVTX's memory extension lets sub-allocators describe their blocks, so when
# running under compute-sanitizer, segments are registered as heaps and blocks as regions
# of them. that doesn't cover everything: copies and memsets that overrun a block, and
# stale pointers to a block that has been reused, still go unnoticed, so allocating every
# array with `cuMemAlloc` (`JULIA_CUDA_MEMORY_POOL=none`) remains the thorough option.
#
# the extension's functions are provided by the sanitizer's injection library, which fills
# a table of function pointers when initializing the extension (see NVIDIA's
# nvtxDetail/nvtxExtInit.h and nvToolsExtMem.h).

struct NvtxExtModuleSegment
  segmentId::Csize_t
  slotCount::Csize_t
  functionSlots::Ptr{Int}
end

struct NvtxExtModuleInfo
  nvtxVer::UInt16
  structSize::UInt16
  moduleId::UInt16
  compatId::UInt16
  segmentsCount::Csize_t
  segments::Ptr{NvtxExtModuleSegment}
  getExportFunction::Ptr{Cvoid}
  extInfo::Ptr{Cvoid}
end

struct NvtxMemVirtualRangeDesc
  size::Csize_t
  ptr::Ptr{Cvoid}
end

struct NvtxMemHeapDesc
  extCompatID::UInt16
  structSize::UInt16
  reserved0::UInt32
  usage::UInt32
  type::UInt32
  typeSpecificDescSize::Csize_t
  typeSpecificDesc::Ptr{Cvoid}
  category::UInt32
  messageType::UInt32
  message::Ptr{Cvoid}
end

struct NvtxMemRegionsRegisterBatch
  extCompatID::UInt16
  structSize::UInt16
  regionType::UInt32
  heap::Ptr{Cvoid}
  regionCount::Csize_t
  regionDescElementSize::Csize_t
  regionDescElements::Ptr{Cvoid}
  regionHandleElementsOut::Ptr{Ptr{Cvoid}}
end

struct NvtxMemRegionsUnregisterBatch
  extCompatID::UInt16
  structSize::UInt16
  refType::UInt32
  refCount::Csize_t
  refElementSize::Csize_t
  refElements::Ptr{Ptr{Cvoid}}
end

const NVTX_VERSION = 3
const NVTX_EXT_MODULEID_MEM = 1
const NVTX_EXT_COMPATID_MEM = 0x0102
const NVTX_MEM_SLOTS = 63
# function slots
const NVTX_MEM_HEAP_REGISTER = 0
const NVTX_MEM_HEAP_UNREGISTER = 1
const NVTX_MEM_HEAP_RESET = 2
const NVTX_MEM_REGIONS_REGISTER = 3
const NVTX_MEM_REGIONS_UNREGISTER = 5
const NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0x1
const NVTX_MEM_TYPE_VIRTUAL_ADDRESS = 0x1
const NVTX_MEM_REGION_REF_TYPE_POINTER = 0x1

# filled by the injection library; slots that hold 0 or 1 aren't available
const nvtx_mem_slots = Base.zeros(Int, NVTX_MEM_SLOTS)

# whether to annotate, i.e., whether running under compute-sanitizer with NVTX support.
# (Nsight Systems also injects NVTX, but doesn't use these annotations)
function annotating()
  @memoize begin
    haskey(ENV, "NV_SANITIZER_INJECTION_PORT_BASE") && initialize_nvtx_mem()
  end::Bool
end

function initialize_nvtx_mem()
  path = get(ENV, "NVTX_INJECTION64_PATH", "")
  isempty(path) && return false
  lib = Libdl.dlopen(path, Libdl.RTLD_NOW; throw_error=false)
  lib === nothing && return false
  init = Libdl.dlsym(lib, :InitializeInjectionNvtxExtension; throw_error=false)
  init === nothing && return false
  segment = Ref(NvtxExtModuleSegment(1, NVTX_MEM_SLOTS, pointer(nvtx_mem_slots)))
  status = GC.@preserve segment begin
    info = Ref(NvtxExtModuleInfo(NVTX_VERSION, sizeof(NvtxExtModuleInfo),
                                 NVTX_EXT_MODULEID_MEM, NVTX_EXT_COMPATID_MEM, 1,
                                 Base.unsafe_convert(Ptr{NvtxExtModuleSegment}, segment),
                                 C_NULL, C_NULL))
    ccall(init, Cint, (Ref{NvtxExtModuleInfo},), info)
  end
  status == 0 && return false
  return all(slot -> nvtx_mem_slots[slot+1] > 1,
             (NVTX_MEM_HEAP_REGISTER, NVTX_MEM_HEAP_UNREGISTER, NVTX_MEM_HEAP_RESET,
              NVTX_MEM_REGIONS_REGISTER, NVTX_MEM_REGIONS_UNREGISTER))
end

nvtx_mem_function(slot) = Ptr{Cvoid}(nvtx_mem_slots[slot+1])

# register a segment, returning a handle for it
function annotate_segment(base::UInt, size::Int)
  range = Ref(NvtxMemVirtualRangeDesc(size, Ptr{Cvoid}(base)))
  GC.@preserve range begin
    desc = Ref(NvtxMemHeapDesc(NVTX_EXT_COMPATID_MEM, sizeof(NvtxMemHeapDesc), 0,
                               NVTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR,
                               NVTX_MEM_TYPE_VIRTUAL_ADDRESS,
                               sizeof(NvtxMemVirtualRangeDesc),
                               Base.unsafe_convert(Ptr{NvtxMemVirtualRangeDesc}, range),
                               0, 0, C_NULL))
    ccall(nvtx_mem_function(NVTX_MEM_HEAP_REGISTER), Ptr{Cvoid},
          (Ptr{Cvoid}, Ref{NvtxMemHeapDesc}), C_NULL, desc)
  end
end

# forget about a segment and its blocks, before releasing it
function annotate_segment_release(heap::Ptr{Cvoid})
  ccall(nvtx_mem_function(NVTX_MEM_HEAP_RESET), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
        C_NULL, heap)
  ccall(nvtx_mem_function(NVTX_MEM_HEAP_UNREGISTER), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}),
        C_NULL, heap)
end

# a block of `size` bytes has been allocated at `addr` in a segment
function annotate_allocation(heap::Ptr{Cvoid}, addr::UInt, size::Int)
  region = Ref(NvtxMemVirtualRangeDesc(size, Ptr{Cvoid}(addr)))
  GC.@preserve region begin
    batch = Ref(NvtxMemRegionsRegisterBatch(NVTX_EXT_COMPATID_MEM,
                                            sizeof(NvtxMemRegionsRegisterBatch),
                                            NVTX_MEM_TYPE_VIRTUAL_ADDRESS, heap, 1,
                                            sizeof(NvtxMemVirtualRangeDesc),
                                            Base.unsafe_convert(Ptr{NvtxMemVirtualRangeDesc}, region),
                                            C_NULL))
    ccall(nvtx_mem_function(NVTX_MEM_REGIONS_REGISTER), Cvoid,
          (Ptr{Cvoid}, Ref{NvtxMemRegionsRegisterBatch}), C_NULL, batch)
  end
  return
end

# the block at `addr` has been freed
function annotate_free(addr::UInt)
  ref = Ref(Ptr{Cvoid}(addr))
  GC.@preserve ref begin
    batch = Ref(NvtxMemRegionsUnregisterBatch(NVTX_EXT_COMPATID_MEM,
                                              sizeof(NvtxMemRegionsUnregisterBatch),
                                              NVTX_MEM_REGION_REF_TYPE_POINTER, 1,
                                              sizeof(Ptr{Cvoid}),
                                              Base.unsafe_convert(Ptr{Ptr{Cvoid}}, ref)))
    ccall(nvtx_mem_function(NVTX_MEM_REGIONS_UNREGISTER), Cvoid,
          (Ptr{Cvoid}, Ref{NvtxMemRegionsUnregisterBatch}), C_NULL, batch)
  end
  return
end

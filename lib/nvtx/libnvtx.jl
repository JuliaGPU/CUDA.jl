using CEnum

initialize_context() = return

struct payload_t
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{payload_t}, f::Symbol)
    f === :ullValue && return Ptr{UInt64}(x + 0)
    f === :llValue && return Ptr{Int64}(x + 0)
    f === :dValue && return Ptr{Cdouble}(x + 0)
    f === :uiValue && return Ptr{UInt32}(x + 0)
    f === :iValue && return Ptr{Int32}(x + 0)
    f === :fValue && return Ptr{Cfloat}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::payload_t, f::Symbol)
    r = Ref{payload_t}(x)
    ptr = Base.unsafe_convert(Ptr{payload_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{payload_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvtxMessageValue_t
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{nvtxMessageValue_t}, f::Symbol)
    f === :ascii && return Ptr{Cstring}(x + 0)
    f === :unicode && return Ptr{Ptr{Cwchar_t}}(x + 0)
    f === :registered && return Ptr{nvtxStringHandle_t}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::nvtxMessageValue_t, f::Symbol)
    r = Ref{nvtxMessageValue_t}(x)
    ptr = Base.unsafe_convert(Ptr{nvtxMessageValue_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvtxMessageValue_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvtxEventAttributes_v2
    data::NTuple{48,UInt8}
end

function Base.getproperty(x::Ptr{nvtxEventAttributes_v2}, f::Symbol)
    f === :version && return Ptr{UInt16}(x + 0)
    f === :size && return Ptr{UInt16}(x + 2)
    f === :category && return Ptr{UInt32}(x + 4)
    f === :colorType && return Ptr{Int32}(x + 8)
    f === :color && return Ptr{UInt32}(x + 12)
    f === :payloadType && return Ptr{Int32}(x + 16)
    f === :reserved0 && return Ptr{Int32}(x + 20)
    f === :payload && return Ptr{payload_t}(x + 24)
    f === :messageType && return Ptr{Int32}(x + 32)
    f === :message && return Ptr{nvtxMessageValue_t}(x + 40)
    return getfield(x, f)
end

function Base.getproperty(x::nvtxEventAttributes_v2, f::Symbol)
    r = Ref{nvtxEventAttributes_v2}(x)
    ptr = Base.unsafe_convert(Ptr{nvtxEventAttributes_v2}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvtxEventAttributes_v2}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

const nvtxEventAttributes_t = nvtxEventAttributes_v2

struct identifier_t
    data::NTuple{8,UInt8}
end

function Base.getproperty(x::Ptr{identifier_t}, f::Symbol)
    f === :pValue && return Ptr{Ptr{Cvoid}}(x + 0)
    f === :ullValue && return Ptr{UInt64}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::identifier_t, f::Symbol)
    r = Ref{identifier_t}(x)
    ptr = Base.unsafe_convert(Ptr{identifier_t}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{identifier_t}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

struct nvtxResourceAttributes_v0
    data::NTuple{32,UInt8}
end

function Base.getproperty(x::Ptr{nvtxResourceAttributes_v0}, f::Symbol)
    f === :version && return Ptr{UInt16}(x + 0)
    f === :size && return Ptr{UInt16}(x + 2)
    f === :identifierType && return Ptr{Int32}(x + 4)
    f === :identifier && return Ptr{identifier_t}(x + 8)
    f === :messageType && return Ptr{Int32}(x + 16)
    f === :message && return Ptr{nvtxMessageValue_t}(x + 24)
    return getfield(x, f)
end

function Base.getproperty(x::nvtxResourceAttributes_v0, f::Symbol)
    r = Ref{nvtxResourceAttributes_v0}(x)
    ptr = Base.unsafe_convert(Ptr{nvtxResourceAttributes_v0}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{nvtxResourceAttributes_v0}, f::Symbol, v)
    return unsafe_store!(getproperty(x, f), v)
end

function nvtxMarkA(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxMarkA(message::Cstring)::Cvoid
end

const nvtxRangeId_t = UInt64

function nvtxRangeStartA(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangeStartA(message::Cstring)::nvtxRangeId_t
end

function nvtxRangePushA(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangePushA(message::Cstring)::Cint
end

function nvtxNameCategoryA(category, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCategoryA(category::UInt32, name::Cstring)::Cvoid
end

function nvtxNameOsThreadA(threadId, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameOsThreadA(threadId::UInt32, name::Cstring)::Cvoid
end

mutable struct nvtxDomainRegistration_st end

const nvtxDomainRegistration = nvtxDomainRegistration_st

const nvtxDomainHandle_t = Ptr{nvtxDomainRegistration}

function nvtxDomainCreateA(name)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainCreateA(name::Cstring)::nvtxDomainHandle_t
end

mutable struct nvtxStringRegistration_st end

const nvtxStringRegistration = nvtxStringRegistration_st

const nvtxStringHandle_t = Ptr{nvtxStringRegistration}

function nvtxDomainRegisterStringA(domain, string)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRegisterStringA(domain::nvtxDomainHandle_t,
                                                   string::Cstring)::nvtxStringHandle_t
end

function nvtxDomainNameCategoryA(domain, category, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainNameCategoryA(domain::nvtxDomainHandle_t,
                                                 category::UInt32, name::Cstring)::Cvoid
end

@cenum nvtxColorType_t::UInt32 begin
    NVTX_COLOR_UNKNOWN = 0
    NVTX_COLOR_ARGB = 1
end

@cenum nvtxMessageType_t::UInt32 begin
    NVTX_MESSAGE_UNKNOWN = 0
    NVTX_MESSAGE_TYPE_ASCII = 1
    NVTX_MESSAGE_TYPE_UNICODE = 2
    NVTX_MESSAGE_TYPE_REGISTERED = 3
end

function nvtxInitialize(reserved)
    initialize_context()
    @ccall libnvtoolsext.nvtxInitialize(reserved::Ptr{Cvoid})::Cvoid
end

@cenum nvtxPayloadType_t::UInt32 begin
    NVTX_PAYLOAD_UNKNOWN = 0
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1
    NVTX_PAYLOAD_TYPE_INT64 = 2
    NVTX_PAYLOAD_TYPE_DOUBLE = 3
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4
    NVTX_PAYLOAD_TYPE_INT32 = 5
    NVTX_PAYLOAD_TYPE_FLOAT = 6
end

function nvtxDomainMarkEx(domain, eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainMarkEx(domain::nvtxDomainHandle_t,
                                          eventAttrib::Ptr{nvtxEventAttributes_t})::Cvoid
end

function nvtxMarkEx(eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxMarkEx(eventAttrib::Ptr{nvtxEventAttributes_t})::Cvoid
end

function nvtxMarkW(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxMarkW(message::Ptr{Cwchar_t})::Cvoid
end

function nvtxDomainRangeStartEx(domain, eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRangeStartEx(domain::nvtxDomainHandle_t,
                                                eventAttrib::Ptr{nvtxEventAttributes_t})::nvtxRangeId_t
end

function nvtxRangeStartEx(eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangeStartEx(eventAttrib::Ptr{nvtxEventAttributes_t})::nvtxRangeId_t
end

function nvtxRangeStartW(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangeStartW(message::Ptr{Cwchar_t})::nvtxRangeId_t
end

function nvtxDomainRangeEnd(domain, id)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRangeEnd(domain::nvtxDomainHandle_t,
                                            id::nvtxRangeId_t)::Cvoid
end

function nvtxRangeEnd(id)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangeEnd(id::nvtxRangeId_t)::Cvoid
end

function nvtxDomainRangePushEx(domain, eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRangePushEx(domain::nvtxDomainHandle_t,
                                               eventAttrib::Ptr{nvtxEventAttributes_t})::Cint
end

function nvtxRangePushEx(eventAttrib)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangePushEx(eventAttrib::Ptr{nvtxEventAttributes_t})::Cint
end

function nvtxRangePushW(message)
    initialize_context()
    @ccall libnvtoolsext.nvtxRangePushW(message::Ptr{Cwchar_t})::Cint
end

function nvtxDomainRangePop(domain)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRangePop(domain::nvtxDomainHandle_t)::Cint
end

function nvtxRangePop()
    initialize_context()
    @ccall libnvtoolsext.nvtxRangePop()::Cint
end

@cenum nvtxResourceGenericType_t::UInt32 begin
    NVTX_RESOURCE_TYPE_UNKNOWN = 0
    NVTX_RESOURCE_TYPE_GENERIC_POINTER = 65537
    NVTX_RESOURCE_TYPE_GENERIC_HANDLE = 65538
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = 65539
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = 65540
end

const nvtxResourceAttributes_t = nvtxResourceAttributes_v0

mutable struct nvtxResourceHandle end

const nvtxResourceHandle_t = Ptr{nvtxResourceHandle}

function nvtxDomainResourceCreate(domain, attribs)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainResourceCreate(domain::nvtxDomainHandle_t,
                                                  attribs::Ptr{nvtxResourceAttributes_t})::nvtxResourceHandle_t
end

function nvtxDomainResourceDestroy(resource)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainResourceDestroy(resource::nvtxResourceHandle_t)::Cvoid
end

function nvtxDomainNameCategoryW(domain, category, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainNameCategoryW(domain::nvtxDomainHandle_t,
                                                 category::UInt32,
                                                 name::Ptr{Cwchar_t})::Cvoid
end

function nvtxNameCategoryW(category, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCategoryW(category::UInt32, name::Ptr{Cwchar_t})::Cvoid
end

function nvtxNameOsThreadW(threadId, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameOsThreadW(threadId::UInt32, name::Ptr{Cwchar_t})::Cvoid
end

function nvtxDomainRegisterStringW(domain, string)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainRegisterStringW(domain::nvtxDomainHandle_t,
                                                   string::Ptr{Cwchar_t})::nvtxStringHandle_t
end

function nvtxDomainCreateW(name)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainCreateW(name::Ptr{Cwchar_t})::nvtxDomainHandle_t
end

function nvtxDomainDestroy(domain)
    initialize_context()
    @ccall libnvtoolsext.nvtxDomainDestroy(domain::nvtxDomainHandle_t)::Cvoid
end

function nvtxNameCuDeviceA(device, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuDeviceA(device::CUdevice, name::Cstring)::Cvoid
end

function nvtxNameCuContextA(context, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuContextA(context::CUcontext, name::Cstring)::Cvoid
end

function nvtxNameCuStreamA(stream, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuStreamA(stream::CUstream, name::Cstring)::Cvoid
end

function nvtxNameCuEventA(event, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuEventA(event::CUevent, name::Cstring)::Cvoid
end

@cenum nvtxResourceCUDAType_t::UInt32 begin
    NVTX_RESOURCE_TYPE_CUDA_DEVICE = 262145
    NVTX_RESOURCE_TYPE_CUDA_CONTEXT = 262146
    NVTX_RESOURCE_TYPE_CUDA_STREAM = 262147
    NVTX_RESOURCE_TYPE_CUDA_EVENT = 262148
end

function nvtxNameCuDeviceW(device, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuDeviceW(device::CUdevice, name::Ptr{Cwchar_t})::Cvoid
end

function nvtxNameCuContextW(context, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuContextW(context::CUcontext, name::Ptr{Cwchar_t})::Cvoid
end

function nvtxNameCuStreamW(stream, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuStreamW(stream::CUstream, name::Ptr{Cwchar_t})::Cvoid
end

function nvtxNameCuEventW(event, name)
    initialize_context()
    @ccall libnvtoolsext.nvtxNameCuEventW(event::CUevent, name::Ptr{Cwchar_t})::Cvoid
end

const NVTX_VERSION = 3

const NVTX_SUCCESS = 0

const NVTX_FAIL = 1

const NVTX_ERR_INIT_LOAD_PROPERTY = 2

const NVTX_ERR_INIT_ACCESS_LIBRARY = 3

const NVTX_ERR_INIT_LOAD_LIBRARY = 4

const NVTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT = 5

const NVTX_ERR_INIT_FAILED_LIBRARY_ENTRY_POINT = 6

const NVTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE = 7

# Skipping MacroDefinition: NVTX_EVENT_ATTRIB_STRUCT_SIZE ( ( uint16_t ) ( sizeof ( nvtxEventAttributes_t ) ) )

const NVTX_RESOURCE_CLASS_GENERIC = 1

# Skipping MacroDefinition: NVTX_RESOURCE_ATTRIB_STRUCT_SIZE ( ( uint16_t ) ( sizeof ( nvtxResourceAttributes_v0 ) ) )

const NVTX_RESOURCE_CLASS_CUDA = 4

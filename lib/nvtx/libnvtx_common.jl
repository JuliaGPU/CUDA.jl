# Automatically generated using Clang.jl

const NVTX_VERSION = 3

# Skipping MacroDefinition: NVTX_INLINE_STATIC inline static

# Skipping MacroDefinition: NVTX_DECLSPEC

# Skipping MacroDefinition: NVTX_VERSIONED_IDENTIFIER_L3 ( NAME , VERSION ) NAME ## _v ## VERSION
# Skipping MacroDefinition: NVTX_VERSIONED_IDENTIFIER_L2 ( NAME , VERSION ) NVTX_VERSIONED_IDENTIFIER_L3 ( NAME , VERSION )
# Skipping MacroDefinition: NVTX_VERSIONED_IDENTIFIER ( NAME ) NVTX_VERSIONED_IDENTIFIER_L2 ( NAME , NVTX_VERSION )

const NVTX_SUCCESS = 0
const NVTX_FAIL = 1
const NVTX_ERR_INIT_LOAD_PROPERTY = 2
const NVTX_ERR_INIT_ACCESS_LIBRARY = 3
const NVTX_ERR_INIT_LOAD_LIBRARY = 4
const NVTX_ERR_INIT_MISSING_LIBRARY_ENTRY_POINT = 5
const NVTX_ERR_INIT_FAILED_LIBRARY_ENTRY_POINT = 6
const NVTX_ERR_NO_INJECTION_LIBRARY_AVAILABLE = 7

# Skipping MacroDefinition: NVTX_EVENT_ATTRIB_STRUCT_SIZE ( ( uint16_t ) ( sizeof ( nvtxEventAttributes_t ) ) )
# Skipping MacroDefinition: NVTX_NO_PUSH_POP_TRACKING ( ( int ) - 2 )
# Skipping MacroDefinition: NVTX_RESOURCE_MAKE_TYPE ( CLASS , INDEX ) ( ( ( ( uint32_t ) ( NVTX_RESOURCE_CLASS_ ## CLASS ) ) << 16 ) | ( ( uint32_t ) ( INDEX ) ) )

const NVTX_RESOURCE_CLASS_GENERIC = 1

# Skipping MacroDefinition: NVTX_RESOURCE_ATTRIB_STRUCT_SIZE ( ( uint16_t ) ( sizeof ( nvtxResourceAttributes_v0 ) ) )

const nvtxRangeId_t = UInt64
const nvtxDomainRegistration_st = Cvoid
const nvtxDomainRegistration = nvtxDomainRegistration_st
const nvtxDomainHandle_t = Ptr{nvtxDomainRegistration}
const nvtxStringRegistration_st = Cvoid
const nvtxStringRegistration = nvtxStringRegistration_st
const nvtxStringHandle_t = Ptr{nvtxStringRegistration}

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

struct nvtxMessageValue_t
    ascii::Cstring
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

struct payload_t
    ullValue::UInt64
end

struct nvtxEventAttributes_v2
    version::UInt16
    size::UInt16
    category::UInt32
    colorType::Int32
    color::UInt32
    payloadType::Int32
    reserved0::Int32
    payload::payload_t
    messageType::Int32
    message::nvtxMessageValue_t
end

const nvtxEventAttributes_t = nvtxEventAttributes_v2

@cenum nvtxResourceGenericType_t::UInt32 begin
    NVTX_RESOURCE_TYPE_UNKNOWN = 0
    NVTX_RESOURCE_TYPE_GENERIC_POINTER = 65537
    NVTX_RESOURCE_TYPE_GENERIC_HANDLE = 65538
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = 65539
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = 65540
end

struct identifier_t
    ullValue::UInt64
end

struct nvtxResourceAttributes_v0
    version::UInt16
    size::UInt16
    identifierType::Int32
    identifier::identifier_t
    messageType::Int32
    message::nvtxMessageValue_t
end

const nvtxResourceAttributes_t = nvtxResourceAttributes_v0
const nvtxResourceHandle = Cvoid
const nvtxResourceHandle_t = Ptr{nvtxResourceHandle}
const NVTX_RESOURCE_CLASS_CUDA = 4

@cenum nvtxResourceCUDAType_t::UInt32 begin
    NVTX_RESOURCE_TYPE_CUDA_DEVICE = 262145
    NVTX_RESOURCE_TYPE_CUDA_CONTEXT = 262146
    NVTX_RESOURCE_TYPE_CUDA_STREAM = 262147
    NVTX_RESOURCE_TYPE_CUDA_EVENT = 262148
end


# Basic library functionality

# TODO: put error here to avoid need for forward decl

#
# API versioning
#

const mapping = Dict{Symbol,Symbol}()
const minreq = Dict{Symbol,VersionNumber}()

if libcuda_version >= v"3.2"
    mapping[:cuDeviceTotalMem]           = :cuDeviceTotalMem_v2
    mapping[:cuCtxCreate]                = :cuCtxCreate_v2
    mapping[:cuModuleGetGlobal]          = :cuModuleGetGlobal_v2
    mapping[:cuMemGetInfo]               = :cuMemGetInfo_v2
    mapping[:cuMemAlloc]                 = :cuMemAlloc_v2
    mapping[:cuMemAllocPitch]            = :cuMemAllocPitch_v2
    mapping[:cuMemFree]                  = :cuMemFree_v2
    mapping[:cuMemGetAddressRange]       = :cuMemGetAddressRange_v2
    mapping[:cuMemAllocHost]             = :cuMemAllocHost_v2
    mapping[:cuMemHostGetDevicePointer]  = :cuMemHostGetDevicePointer_v2
    mapping[:cuMemcpyHtoD]               = :cuMemcpyHtoD_v2
    mapping[:cuMemcpyDtoH]               = :cuMemcpyDtoH_v2
    mapping[:cuMemcpyDtoD]               = :cuMemcpyDtoD_v2
    mapping[:cuMemcpyDtoA]               = :cuMemcpyDtoA_v2
    mapping[:cuMemcpyAtoD]               = :cuMemcpyAtoD_v2
    mapping[:cuMemcpyHtoA]               = :cuMemcpyHtoA_v2
    mapping[:cuMemcpyAtoH]               = :cuMemcpyAtoH_v2
    mapping[:cuMemcpyAtoA]               = :cuMemcpyAtoA_v2
    mapping[:cuMemcpyHtoAAsync]          = :cuMemcpyHtoAAsync_v2
    mapping[:cuMemcpyAtoHAsync]          = :cuMemcpyAtoHAsync_v2
    mapping[:cuMemcpy2D]                 = :cuMemcpy2D_v2
    mapping[:cuMemcpy2DUnaligned]        = :cuMemcpy2DUnaligned_v2
    mapping[:cuMemcpy3D]                 = :cuMemcpy3D_v2
    mapping[:cuMemcpyHtoDAsync]          = :cuMemcpyHtoDAsync_v2
    mapping[:cuMemcpyDtoHAsync]          = :cuMemcpyDtoHAsync_v2
    mapping[:cuMemcpyDtoDAsync]          = :cuMemcpyDtoDAsync_v2
    mapping[:cuMemcpy2DAsync]            = :cuMemcpy2DAsync_v2
    mapping[:cuMemcpy3DAsync]            = :cuMemcpy3DAsync_v2
    mapping[:cuMemsetD8]                 = :cuMemsetD8_v2
    mapping[:cuMemsetD16]                = :cuMemsetD16_v2
    mapping[:cuMemsetD32]                = :cuMemsetD32_v2
    mapping[:cuMemsetD2D8]               = :cuMemsetD2D8_v2
    mapping[:cuMemsetD2D16]              = :cuMemsetD2D16_v2
    mapping[:cuMemsetD2D32]              = :cuMemsetD2D32_v2
    mapping[:cuArrayCreate]              = :cuArrayCreate_v2
    mapping[:cuArrayGetDescriptor]       = :cuArrayGetDescriptor_v2
    mapping[:cuArray3DCreate]            = :cuArray3DCreate_v2
    mapping[:cuArray3DGetDescriptor]     = :cuArray3DGetDescriptor_v2
    mapping[:cuTexRefSetAddress]         = :cuTexRefSetAddress_v2
    mapping[:cuTexRefGetAddress]         = :cuTexRefGetAddress_v2
    mapping[:cuGraphicsResourceGetMappedPointer] = :cuGraphicsResourceGetMappedPointer_v2
end

if libcuda_version >= v"4.0"
    mapping[:cuCtxDestroy]               = :cuCtxDestroy_v2
    mapping[:cuCtxPopCurrent]            = :cuCtxPopCurrent_v2
    mapping[:cuCtxPushCurrent]           = :cuCtxPushCurrent_v2
    mapping[:cuStreamDestroy]            = :cuStreamDestroy_v2
    mapping[:cuEventDestroy]             = :cuEventDestroy_v2
end

if libcuda_version >= v"4.1"
    mapping[:cuTexRefSetAddress2D]       = :cuTexRefSetAddress2D_v3
end

if libcuda_version >= v"6.5"
    mapping[:cuLinkCreate]              = :cuLinkCreate_v2
    mapping[:cuLinkAddData]             = :cuLinkAddData_v2
    mapping[:cuLinkAddFile]             = :cuLinkAddFile_v2
end

if libcuda_version >= v"6.5"
    mapping[:cuMemHostRegister]         = :cuMemHostRegister_v2
    mapping[:cuGraphicsResourceSetMapFlags] = :cuGraphicsResourceSetMapFlags_v2
end

if v"3.2" <= libcuda_version < v"4.1"
    mapping[:cuTexRefSetAddress2D]      = :cuTexRefSetAddress2D_v2
end


## Version-dependent features

minreq[:cuLinkCreate]       = v"5.5"
minreq[:cuLinkDestroy]      = v"5.5"
minreq[:cuLinkComplete]     = v"5.5"
minreq[:cuLinkAddFile]      = v"5.5"
minreq[:cuLinkAddData]      = v"5.5"

minreq[:cuDummyAvailable]   = v"0"      # non-existing functions
minreq[:cuDummyUnavailable] = v"999"    # for testing purposes

# explicitly mark unavailable symbols, signaling `@apicall` to error out
for (api_function, minimum_version) in minreq
    if libcuda_version < minimum_version
        mapping[api_function] = :unavailable
    end
end


#
# API call wrapper
#

# forward definition of CuError
const CuError_t = Cint
"""
    CuError(code::Integer)
    CuError(code::Integer, info::String)

Create a CUDA error object with error code `code`. The optional `info` parameter indicates
whether extra information, such as error logs, is known.
"""
struct CuError <: Exception
    code::CuError_t
    meta::Any

    CuError(code, meta=nothing) = new(code, meta)
end

# ccall wrapper for calling functions in NVIDIA libraries
const apicall_hook = Ref{Union{Nothing,Function}}(nothing)
macro apicall(funspec, argtypes, args...)
    isa(funspec, QuoteNode) || error("first argument to @apicall should be a symbol")
    fun = funspec.value

    # resolve the function
    global mapping, version_requirements
    api_fun = get(mapping, fun, fun)
    if api_fun == :unavailable
        return :(throw(CuVersionError($(QuoteNode(fun)), $(minreq[fun]))))
    end

    return quote
        # NOTE: this hook is used by CUDAnative.jl to initialize upon the first API call
        apicall_hook[] !== nothing && apicall_hook[]($funspec)

        status = ccall(($(QuoteNode(api_fun)), libcuda), CuError_t,
                       $(esc(argtypes)), $(map(esc, args)...))

        if status != SUCCESS.code
            err = CuError(status)
            throw(err)
        end
    end
end

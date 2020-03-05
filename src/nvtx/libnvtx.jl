# Julia wrapper for header: nvToolsExt.h
# Automatically generated using Clang.jl


function nvtxInitialize(reserved)
    @runtime_ccall((:nvtxInitialize, libnvtx()), Cvoid,
          (Ptr{Cvoid},),
          reserved)
end

function nvtxDomainMarkEx(domain, eventAttrib)
    @runtime_ccall((:nvtxDomainMarkEx, libnvtx()), Cvoid,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxMarkEx(eventAttrib)
    @runtime_ccall((:nvtxMarkEx, libnvtx()), Cvoid,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxMarkA(message)
    @runtime_ccall((:nvtxMarkA, libnvtx()), Cvoid,
          (Cstring,),
          message)
end

function nvtxMarkW(message)
    @runtime_ccall((:nvtxMarkW, libnvtx()), Cvoid,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangeStartEx(domain, eventAttrib)
    @runtime_ccall((:nvtxDomainRangeStartEx, libnvtx()), nvtxRangeId_t,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxRangeStartEx(eventAttrib)
    @runtime_ccall((:nvtxRangeStartEx, libnvtx()), nvtxRangeId_t,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxRangeStartA(message)
    @runtime_ccall((:nvtxRangeStartA, libnvtx()), nvtxRangeId_t,
          (Cstring,),
          message)
end

function nvtxRangeStartW(message)
    @runtime_ccall((:nvtxRangeStartW, libnvtx()), nvtxRangeId_t,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangeEnd(domain, id)
    @runtime_ccall((:nvtxDomainRangeEnd, libnvtx()), Cvoid,
          (nvtxDomainHandle_t, nvtxRangeId_t),
          domain, id)
end

function nvtxRangeEnd(id)
    @runtime_ccall((:nvtxRangeEnd, libnvtx()), Cvoid,
          (nvtxRangeId_t,),
          id)
end

function nvtxDomainRangePushEx(domain, eventAttrib)
    @runtime_ccall((:nvtxDomainRangePushEx, libnvtx()), Cint,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxRangePushEx(eventAttrib)
    @runtime_ccall((:nvtxRangePushEx, libnvtx()), Cint,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxRangePushA(message)
    @runtime_ccall((:nvtxRangePushA, libnvtx()), Cint,
          (Cstring,),
          message)
end

function nvtxRangePushW(message)
    @runtime_ccall((:nvtxRangePushW, libnvtx()), Cint,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangePop(domain)
    @runtime_ccall((:nvtxDomainRangePop, libnvtx()), Cint,
          (nvtxDomainHandle_t,),
          domain)
end

function nvtxRangePop()
    @runtime_ccall((:nvtxRangePop, libnvtx()), Cint, ())
end

function nvtxDomainResourceCreate(domain, attribs)
    @runtime_ccall((:nvtxDomainResourceCreate, libnvtx()), nvtxResourceHandle_t,
          (nvtxDomainHandle_t, Ptr{nvtxResourceAttributes_t}),
          domain, attribs)
end

function nvtxDomainResourceDestroy(resource)
    @runtime_ccall((:nvtxDomainResourceDestroy, libnvtx()), Cvoid,
          (nvtxResourceHandle_t,),
          resource)
end

function nvtxDomainNameCategoryA(domain, category, name)
    @runtime_ccall((:nvtxDomainNameCategoryA, libnvtx()), Cvoid,
          (nvtxDomainHandle_t, UInt32, Cstring),
          domain, category, name)
end

function nvtxDomainNameCategoryW(domain, category, name)
    @runtime_ccall((:nvtxDomainNameCategoryW, libnvtx()), Cvoid,
          (nvtxDomainHandle_t, UInt32, Ptr{Cwchar_t}),
          domain, category, name)
end

function nvtxNameCategoryA(category, name)
    @runtime_ccall((:nvtxNameCategoryA, libnvtx()), Cvoid,
          (UInt32, Cstring),
          category, name)
end

function nvtxNameCategoryW(category, name)
    @runtime_ccall((:nvtxNameCategoryW, libnvtx()), Cvoid,
          (UInt32, Ptr{Cwchar_t}),
          category, name)
end

function nvtxNameOsThreadA(threadId, name)
    @runtime_ccall((:nvtxNameOsThreadA, libnvtx()), Cvoid,
          (UInt32, Cstring),
          threadId, name)
end

function nvtxNameOsThreadW(threadId, name)
    @runtime_ccall((:nvtxNameOsThreadW, libnvtx()), Cvoid,
          (UInt32, Ptr{Cwchar_t}),
          threadId, name)
end

function nvtxDomainRegisterStringA(domain, string)
    @runtime_ccall((:nvtxDomainRegisterStringA, libnvtx()), nvtxStringHandle_t,
          (nvtxDomainHandle_t, Cstring),
          domain, string)
end

function nvtxDomainRegisterStringW(domain, string)
    @runtime_ccall((:nvtxDomainRegisterStringW, libnvtx()), nvtxStringHandle_t,
          (nvtxDomainHandle_t, Ptr{Cwchar_t}),
          domain, string)
end

function nvtxDomainCreateA(name)
    @runtime_ccall((:nvtxDomainCreateA, libnvtx()), nvtxDomainHandle_t,
          (Cstring,),
          name)
end

function nvtxDomainCreateW(name)
    @runtime_ccall((:nvtxDomainCreateW, libnvtx()), nvtxDomainHandle_t,
          (Ptr{Cwchar_t},),
          name)
end

function nvtxDomainDestroy(domain)
    @runtime_ccall((:nvtxDomainDestroy, libnvtx()), Cvoid,
          (nvtxDomainHandle_t,),
          domain)
end
# Julia wrapper for header: nvToolsExtCuda.h
# Automatically generated using Clang.jl


function nvtxNameCuDeviceA(device, name)
    @runtime_ccall((:nvtxNameCuDeviceA, libnvtx()), Cvoid,
          (CUdevice, Cstring),
          device, name)
end

function nvtxNameCuDeviceW(device, name)
    @runtime_ccall((:nvtxNameCuDeviceW, libnvtx()), Cvoid,
          (CUdevice, Ptr{Cwchar_t}),
          device, name)
end

function nvtxNameCuContextA(context, name)
    @runtime_ccall((:nvtxNameCuContextA, libnvtx()), Cvoid,
          (CUcontext, Cstring),
          context, name)
end

function nvtxNameCuContextW(context, name)
    @runtime_ccall((:nvtxNameCuContextW, libnvtx()), Cvoid,
          (CUcontext, Ptr{Cwchar_t}),
          context, name)
end

function nvtxNameCuStreamA(stream, name)
    @runtime_ccall((:nvtxNameCuStreamA, libnvtx()), Cvoid,
          (CUstream, Cstring),
          stream, name)
end

function nvtxNameCuStreamW(stream, name)
    @runtime_ccall((:nvtxNameCuStreamW, libnvtx()), Cvoid,
          (CUstream, Ptr{Cwchar_t}),
          stream, name)
end

function nvtxNameCuEventA(event, name)
    @runtime_ccall((:nvtxNameCuEventA, libnvtx()), Cvoid,
          (CUevent, Cstring),
          event, name)
end

function nvtxNameCuEventW(event, name)
    @runtime_ccall((:nvtxNameCuEventW, libnvtx()), Cvoid,
          (CUevent, Ptr{Cwchar_t}),
          event, name)
end

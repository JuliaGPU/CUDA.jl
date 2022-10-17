# Julia wrapper for header: nvToolsExt.h
# Automatically generated using Clang.jl

function nvtxInitialize(reserved)
    initialize_context()
    ccall((:nvtxInitialize, libnvtoolsext), Cvoid,
                   (Ptr{Cvoid},),
                   reserved)
end

function nvtxDomainMarkEx(domain, eventAttrib)
    initialize_context()
    ccall((:nvtxDomainMarkEx, libnvtoolsext), Cvoid,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxMarkEx(eventAttrib)
    initialize_context()
    ccall((:nvtxMarkEx, libnvtoolsext), Cvoid,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxMarkA(message)
    initialize_context()
    ccall((:nvtxMarkA, libnvtoolsext), Cvoid,
                   (Cstring,),
                   message)
end

function nvtxMarkW(message)
    initialize_context()
    ccall((:nvtxMarkW, libnvtoolsext), Cvoid,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangeStartEx(domain, eventAttrib)
    initialize_context()
    ccall((:nvtxDomainRangeStartEx, libnvtoolsext), nvtxRangeId_t,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxRangeStartEx(eventAttrib)
    initialize_context()
    ccall((:nvtxRangeStartEx, libnvtoolsext), nvtxRangeId_t,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxRangeStartA(message)
    initialize_context()
    ccall((:nvtxRangeStartA, libnvtoolsext), nvtxRangeId_t,
                   (Cstring,),
                   message)
end

function nvtxRangeStartW(message)
    initialize_context()
    ccall((:nvtxRangeStartW, libnvtoolsext), nvtxRangeId_t,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangeEnd(domain, id)
    initialize_context()
    ccall((:nvtxDomainRangeEnd, libnvtoolsext), Cvoid,
                   (nvtxDomainHandle_t, nvtxRangeId_t),
                   domain, id)
end

function nvtxRangeEnd(id)
    initialize_context()
    ccall((:nvtxRangeEnd, libnvtoolsext), Cvoid,
                   (nvtxRangeId_t,),
                   id)
end

function nvtxDomainRangePushEx(domain, eventAttrib)
    initialize_context()
    ccall((:nvtxDomainRangePushEx, libnvtoolsext), Cint,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxRangePushEx(eventAttrib)
    initialize_context()
    ccall((:nvtxRangePushEx, libnvtoolsext), Cint,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxRangePushA(message)
    initialize_context()
    ccall((:nvtxRangePushA, libnvtoolsext), Cint,
                   (Cstring,),
                   message)
end

function nvtxRangePushW(message)
    initialize_context()
    ccall((:nvtxRangePushW, libnvtoolsext), Cint,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangePop(domain)
    initialize_context()
    ccall((:nvtxDomainRangePop, libnvtoolsext), Cint,
                   (nvtxDomainHandle_t,),
                   domain)
end

function nvtxRangePop()
    initialize_context()
    ccall((:nvtxRangePop, libnvtoolsext), Cint, ())
end

function nvtxDomainResourceCreate(domain, attribs)
    initialize_context()
    ccall((:nvtxDomainResourceCreate, libnvtoolsext), nvtxResourceHandle_t,
                   (nvtxDomainHandle_t, Ptr{nvtxResourceAttributes_t}),
                   domain, attribs)
end

function nvtxDomainResourceDestroy(resource)
    initialize_context()
    ccall((:nvtxDomainResourceDestroy, libnvtoolsext), Cvoid,
                   (nvtxResourceHandle_t,),
                   resource)
end

function nvtxDomainNameCategoryA(domain, category, name)
    initialize_context()
    ccall((:nvtxDomainNameCategoryA, libnvtoolsext), Cvoid,
                   (nvtxDomainHandle_t, UInt32, Cstring),
                   domain, category, name)
end

function nvtxDomainNameCategoryW(domain, category, name)
    initialize_context()
    ccall((:nvtxDomainNameCategoryW, libnvtoolsext), Cvoid,
                   (nvtxDomainHandle_t, UInt32, Ptr{Cwchar_t}),
                   domain, category, name)
end

function nvtxNameCategoryA(category, name)
    initialize_context()
    ccall((:nvtxNameCategoryA, libnvtoolsext), Cvoid,
                   (UInt32, Cstring),
                   category, name)
end

function nvtxNameCategoryW(category, name)
    initialize_context()
    ccall((:nvtxNameCategoryW, libnvtoolsext), Cvoid,
                   (UInt32, Ptr{Cwchar_t}),
                   category, name)
end

function nvtxNameOsThreadA(threadId, name)
    initialize_context()
    ccall((:nvtxNameOsThreadA, libnvtoolsext), Cvoid,
                   (UInt32, Cstring),
                   threadId, name)
end

function nvtxNameOsThreadW(threadId, name)
    initialize_context()
    ccall((:nvtxNameOsThreadW, libnvtoolsext), Cvoid,
                   (UInt32, Ptr{Cwchar_t}),
                   threadId, name)
end

function nvtxDomainRegisterStringA(domain, string)
    initialize_context()
    ccall((:nvtxDomainRegisterStringA, libnvtoolsext), nvtxStringHandle_t,
                   (nvtxDomainHandle_t, Cstring),
                   domain, string)
end

function nvtxDomainRegisterStringW(domain, string)
    initialize_context()
    ccall((:nvtxDomainRegisterStringW, libnvtoolsext), nvtxStringHandle_t,
                   (nvtxDomainHandle_t, Ptr{Cwchar_t}),
                   domain, string)
end

function nvtxDomainCreateA(name)
    initialize_context()
    ccall((:nvtxDomainCreateA, libnvtoolsext), nvtxDomainHandle_t,
                   (Cstring,),
                   name)
end

function nvtxDomainCreateW(name)
    initialize_context()
    ccall((:nvtxDomainCreateW, libnvtoolsext), nvtxDomainHandle_t,
                   (Ptr{Cwchar_t},),
                   name)
end

function nvtxDomainDestroy(domain)
    initialize_context()
    ccall((:nvtxDomainDestroy, libnvtoolsext), Cvoid,
                   (nvtxDomainHandle_t,),
                   domain)
end
# Julia wrapper for header: nvToolsExtCuda.h
# Automatically generated using Clang.jl

function nvtxNameCuDeviceA(device, name)
    initialize_context()
    ccall((:nvtxNameCuDeviceA, libnvtoolsext), Cvoid,
                   (CUdevice, Cstring),
                   device, name)
end

function nvtxNameCuDeviceW(device, name)
    initialize_context()
    ccall((:nvtxNameCuDeviceW, libnvtoolsext), Cvoid,
                   (CUdevice, Ptr{Cwchar_t}),
                   device, name)
end

function nvtxNameCuContextA(context, name)
    initialize_context()
    ccall((:nvtxNameCuContextA, libnvtoolsext), Cvoid,
                   (CUcontext, Cstring),
                   context, name)
end

function nvtxNameCuContextW(context, name)
    initialize_context()
    ccall((:nvtxNameCuContextW, libnvtoolsext), Cvoid,
                   (CUcontext, Ptr{Cwchar_t}),
                   context, name)
end

function nvtxNameCuStreamA(stream, name)
    initialize_context()
    ccall((:nvtxNameCuStreamA, libnvtoolsext), Cvoid,
                   (CUstream, Cstring),
                   stream, name)
end

function nvtxNameCuStreamW(stream, name)
    initialize_context()
    ccall((:nvtxNameCuStreamW, libnvtoolsext), Cvoid,
                   (CUstream, Ptr{Cwchar_t}),
                   stream, name)
end

function nvtxNameCuEventA(event, name)
    initialize_context()
    ccall((:nvtxNameCuEventA, libnvtoolsext), Cvoid,
                   (CUevent, Cstring),
                   event, name)
end

function nvtxNameCuEventW(event, name)
    initialize_context()
    ccall((:nvtxNameCuEventW, libnvtoolsext), Cvoid,
                   (CUevent, Ptr{Cwchar_t}),
                   event, name)
end

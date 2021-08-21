# Julia wrapper for header: nvToolsExt.h
# Automatically generated using Clang.jl

function nvtxInitialize(reserved)
    initialize_api()
    ccall((:nvtxInitialize, libnvtx()), Cvoid,
                   (Ptr{Cvoid},),
                   reserved)
end

function nvtxDomainMarkEx(domain, eventAttrib)
    initialize_api()
    ccall((:nvtxDomainMarkEx, libnvtx()), Cvoid,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxMarkEx(eventAttrib)
    initialize_api()
    ccall((:nvtxMarkEx, libnvtx()), Cvoid,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxMarkA(message)
    initialize_api()
    ccall((:nvtxMarkA, libnvtx()), Cvoid,
                   (Cstring,),
                   message)
end

function nvtxMarkW(message)
    initialize_api()
    ccall((:nvtxMarkW, libnvtx()), Cvoid,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangeStartEx(domain, eventAttrib)
    initialize_api()
    ccall((:nvtxDomainRangeStartEx, libnvtx()), nvtxRangeId_t,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxRangeStartEx(eventAttrib)
    initialize_api()
    ccall((:nvtxRangeStartEx, libnvtx()), nvtxRangeId_t,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxRangeStartA(message)
    initialize_api()
    ccall((:nvtxRangeStartA, libnvtx()), nvtxRangeId_t,
                   (Cstring,),
                   message)
end

function nvtxRangeStartW(message)
    initialize_api()
    ccall((:nvtxRangeStartW, libnvtx()), nvtxRangeId_t,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangeEnd(domain, id)
    initialize_api()
    ccall((:nvtxDomainRangeEnd, libnvtx()), Cvoid,
                   (nvtxDomainHandle_t, nvtxRangeId_t),
                   domain, id)
end

function nvtxRangeEnd(id)
    initialize_api()
    ccall((:nvtxRangeEnd, libnvtx()), Cvoid,
                   (nvtxRangeId_t,),
                   id)
end

function nvtxDomainRangePushEx(domain, eventAttrib)
    initialize_api()
    ccall((:nvtxDomainRangePushEx, libnvtx()), Cint,
                   (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
                   domain, eventAttrib)
end

function nvtxRangePushEx(eventAttrib)
    initialize_api()
    ccall((:nvtxRangePushEx, libnvtx()), Cint,
                   (Ptr{nvtxEventAttributes_t},),
                   eventAttrib)
end

function nvtxRangePushA(message)
    initialize_api()
    ccall((:nvtxRangePushA, libnvtx()), Cint,
                   (Cstring,),
                   message)
end

function nvtxRangePushW(message)
    initialize_api()
    ccall((:nvtxRangePushW, libnvtx()), Cint,
                   (Ptr{Cwchar_t},),
                   message)
end

function nvtxDomainRangePop(domain)
    initialize_api()
    ccall((:nvtxDomainRangePop, libnvtx()), Cint,
                   (nvtxDomainHandle_t,),
                   domain)
end

function nvtxRangePop()
    initialize_api()
    ccall((:nvtxRangePop, libnvtx()), Cint, ())
end

function nvtxDomainResourceCreate(domain, attribs)
    initialize_api()
    ccall((:nvtxDomainResourceCreate, libnvtx()), nvtxResourceHandle_t,
                   (nvtxDomainHandle_t, Ptr{nvtxResourceAttributes_t}),
                   domain, attribs)
end

function nvtxDomainResourceDestroy(resource)
    initialize_api()
    ccall((:nvtxDomainResourceDestroy, libnvtx()), Cvoid,
                   (nvtxResourceHandle_t,),
                   resource)
end

function nvtxDomainNameCategoryA(domain, category, name)
    initialize_api()
    ccall((:nvtxDomainNameCategoryA, libnvtx()), Cvoid,
                   (nvtxDomainHandle_t, UInt32, Cstring),
                   domain, category, name)
end

function nvtxDomainNameCategoryW(domain, category, name)
    initialize_api()
    ccall((:nvtxDomainNameCategoryW, libnvtx()), Cvoid,
                   (nvtxDomainHandle_t, UInt32, Ptr{Cwchar_t}),
                   domain, category, name)
end

function nvtxNameCategoryA(category, name)
    initialize_api()
    ccall((:nvtxNameCategoryA, libnvtx()), Cvoid,
                   (UInt32, Cstring),
                   category, name)
end

function nvtxNameCategoryW(category, name)
    initialize_api()
    ccall((:nvtxNameCategoryW, libnvtx()), Cvoid,
                   (UInt32, Ptr{Cwchar_t}),
                   category, name)
end

function nvtxNameOsThreadA(threadId, name)
    initialize_api()
    ccall((:nvtxNameOsThreadA, libnvtx()), Cvoid,
                   (UInt32, Cstring),
                   threadId, name)
end

function nvtxNameOsThreadW(threadId, name)
    initialize_api()
    ccall((:nvtxNameOsThreadW, libnvtx()), Cvoid,
                   (UInt32, Ptr{Cwchar_t}),
                   threadId, name)
end

function nvtxDomainRegisterStringA(domain, string)
    initialize_api()
    ccall((:nvtxDomainRegisterStringA, libnvtx()), nvtxStringHandle_t,
                   (nvtxDomainHandle_t, Cstring),
                   domain, string)
end

function nvtxDomainRegisterStringW(domain, string)
    initialize_api()
    ccall((:nvtxDomainRegisterStringW, libnvtx()), nvtxStringHandle_t,
                   (nvtxDomainHandle_t, Ptr{Cwchar_t}),
                   domain, string)
end

function nvtxDomainCreateA(name)
    initialize_api()
    ccall((:nvtxDomainCreateA, libnvtx()), nvtxDomainHandle_t,
                   (Cstring,),
                   name)
end

function nvtxDomainCreateW(name)
    initialize_api()
    ccall((:nvtxDomainCreateW, libnvtx()), nvtxDomainHandle_t,
                   (Ptr{Cwchar_t},),
                   name)
end

function nvtxDomainDestroy(domain)
    initialize_api()
    ccall((:nvtxDomainDestroy, libnvtx()), Cvoid,
                   (nvtxDomainHandle_t,),
                   domain)
end
# Julia wrapper for header: nvToolsExtCuda.h
# Automatically generated using Clang.jl

function nvtxNameCuDeviceA(device, name)
    initialize_api()
    ccall((:nvtxNameCuDeviceA, libnvtx()), Cvoid,
                   (CUdevice, Cstring),
                   device, name)
end

function nvtxNameCuDeviceW(device, name)
    initialize_api()
    ccall((:nvtxNameCuDeviceW, libnvtx()), Cvoid,
                   (CUdevice, Ptr{Cwchar_t}),
                   device, name)
end

function nvtxNameCuContextA(context, name)
    initialize_api()
    ccall((:nvtxNameCuContextA, libnvtx()), Cvoid,
                   (CUcontext, Cstring),
                   context, name)
end

function nvtxNameCuContextW(context, name)
    initialize_api()
    ccall((:nvtxNameCuContextW, libnvtx()), Cvoid,
                   (CUcontext, Ptr{Cwchar_t}),
                   context, name)
end

function nvtxNameCuStreamA(stream, name)
    initialize_api()
    ccall((:nvtxNameCuStreamA, libnvtx()), Cvoid,
                   (CUstream, Cstring),
                   stream, name)
end

function nvtxNameCuStreamW(stream, name)
    initialize_api()
    ccall((:nvtxNameCuStreamW, libnvtx()), Cvoid,
                   (CUstream, Ptr{Cwchar_t}),
                   stream, name)
end

function nvtxNameCuEventA(event, name)
    initialize_api()
    ccall((:nvtxNameCuEventA, libnvtx()), Cvoid,
                   (CUevent, Cstring),
                   event, name)
end

function nvtxNameCuEventW(event, name)
    initialize_api()
    ccall((:nvtxNameCuEventW, libnvtx()), Cvoid,
                   (CUevent, Ptr{Cwchar_t}),
                   event, name)
end

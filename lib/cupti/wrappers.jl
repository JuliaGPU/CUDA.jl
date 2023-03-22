function version()
    version_ref = Ref{Cuint}()
    cuptiGetVersion(version_ref)
    VersionNumber(version_ref[])
end

    # params = Ref(CUpti_Profiler_CounterDataImage_CalculateSize_Params(
    #     CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
    #     C_NULL, 0, C_NULL, 0))
    # cuptiProfilerCounterDataImageCalculateSize(params)


function initialize_profiler()
    params = Ref(CUpti_Profiler_Initialize_Params(
        CUpti_Profiler_Initialize_Params_STRUCT_SIZE,
        C_NULL))
    cuptiProfilerInitialize(params)
end

function deinitalize_profiler()
    params = Ref(CUpti_Profiler_DeInitialize_Params(
        CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE,
        C_NULL))
    cuptiProfilerDeInitialize(params)
end

function counter_availability(ctx = context())
    # 1. Query size
    params = Ref(CUpti_Profiler_GetCounterAvailability_Params(
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        C_NULL, ctx.handle, 0, C_NULL))
    cuptiProfilerGetCounterAvailability(params)

    sz = params[].counterAvailabilityImageSize
    buffer = Vector{UInt8}(undef, sz)

    GC.@preserve buffer begin
        params = Ref(CUpti_Profiler_GetCounterAvailability_Params(
        CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE,
        C_NULL, ctx.handle, sz, pointer(buffer)))
        cuptiProfilerGetCounterAvailability(params)
    end
    return buffer
end

abstract type CounterDataBuilder end
function prefix end 

function CounterDataImage(builder, maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength)
    p = prefix(builder)
    GC.@preserve p begin
        options = Ref(CUpti_Profiler_CounterDataImageOptions(
            CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE,
            C_NULL, pointer(p), length(p), maxNumRanges, maxNumRangeTreeNodes, maxRangeNameLength))

        GC.@preserve options begin
            params = Ref(CUpti_Profiler_CounterDataImage_CalculateSize_Params(
                CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE,
                C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, pointer(options), 0))
            CUpti_Profiler_CounterDataImage_CalculateSize(params)
            sz = params[].CounterDataImageSize
        end 
        dataImage = Vector{UInt8}(undef, sz)
        GC.@preserve options dataImage begin
            params = Ref(CUpti_Profiler_CounterDataImage_Initialize_Params(
                CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE,
                C_NULL, CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE, pointer(options),
                sz, pointer(dataImage)))
            CUpti_Profiler_CounterDataImage_Initialize(params)
        end
    end

    GC.@preserve dataImage begin
        params = Ref(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params(
            CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE,
            C_NULL, sz, pointer(dataImage),0))
        CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize(params)
        scratch_sz = params[].counterDataScratchBufferSize
    end
    scratch = Vector{UInt8}(undef, scratch_sz)
    GC.@preserve dataImage scratch begin
        params = Ref(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params(
            CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE,
            C_NULL, sz, pointer(dataImage),scratch_sz, pointer(scratch)))
        CUpti_Profiler_CounterDataImage_InitializeScratchBuffer(params)
    end

    return (; dataImage, scratch)
end


mutable struct CounterDataImage


end


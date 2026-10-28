#
# PC sampling API
#
# Periodically samples the program counter of a warp on every SM, along with the reason
# the warp is stalled (or that it was selected to issue an instruction). Kernels execute
# concurrently and at full speed, while CUPTI aggregates the samples per PC.

"""
    CUPTI.has_pc_sampling() -> Bool

Whether the loaded CUPTI library supports PC sampling as CUDA.jl uses it (CUDA 13 or
later).
"""
has_pc_sampling() = isdefined(@__MODULE__, :libcupti) && version() >= v"13"

"""
    CUPTI.check_pc_sampling_support(dev::CuDevice)

Throw an error explaining why `dev` doesn't support PC sampling, if it doesn't.
"""
check_pc_sampling_support(dev::CuDevice) =
    check_profiler_support(dev; api=CUPTI_PROFILER_PC_SAMPLING, feature="PC sampling")

# a configuration attribute, with its value stored in the attribute's data union
function pc_sampling_attribute(type::CUpti_PCSamplingConfigurationAttributeType, value)
    ref = Ref{CUpti_PCSamplingConfigurationInfo}()
    GC.@preserve ref begin
        ptr = Base.unsafe_convert(Ptr{CUpti_PCSamplingConfigurationInfo}, ref)
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0,
              sizeof(CUpti_PCSamplingConfigurationInfo))
        ptr.attributeType = type
        unsafe_store!(Ptr{typeof(value)}(ptr + 8), value)
    end
    return ref[]
end

function set_pc_sampling_attributes(ctx::CuContext, attrs::Vector{CUpti_PCSamplingConfigurationInfo})
    GC.@preserve attrs begin
        params = profiler_params(CUpti_PCSamplingConfigurationInfoParams,
                                 :pPCSamplingConfigurationInfo; ctx,
                                 numAttributes=length(attrs),
                                 pPCSamplingConfigurationInfo=pointer(attrs))
        cuptiPCSamplingSetConfigurationAttribute(params)
    end
end

# the stall reasons CUPTI reports for the device of `ctx`, by their (arbitrary) index
function stall_reasons(ctx::CuContext)
    count = Ref{Csize_t}(0)
    params = profiler_params(CUpti_PCSamplingGetNumStallReasonsParams, :numStallReasons;
                             ctx, numStallReasons=count)
    GC.@preserve count cuptiPCSamplingGetNumStallReasons(params)
    # CUPTI returns no stall reasons, rather than an error, when it fails to load the
    # NVIDIA performance library (e.g. on Tegra, when libnvperf can't be found)
    count[] == 0 && error("CUPTI did not report any stall reasons; PC sampling is not available on this system.")

    indices = Vector{UInt32}(undef, count[])
    buffers = [zeros(UInt8, CUPTI_STALL_REASON_STRING_SIZE) for _ in 1:count[]]
    names = map(pointer, buffers)
    GC.@preserve buffers names begin
        params = profiler_params(CUpti_PCSamplingGetStallReasonsParams, :stallReasons;
                                 ctx, numStallReasons=count[],
                                 stallReasonIndex=pointer(indices),
                                 stallReasons=pointer(names))
        cuptiPCSamplingGetStallReasons(params)
    end
    Dict(index => unsafe_string(pointer(buffer)) for (index, buffer) in zip(indices, buffers))
end

# a buffer that CUPTI writes PC records to
struct PCSamplingBuffer
    data::Base.RefValue{CUpti_PCSamplingData}
    pcs::Vector{CUpti_PCSamplingPCData}
    # the stall reason counts of each PC, which the records point to
    reasons::Matrix{CUpti_PCSamplingStallReason}
end

function PCSamplingBuffer(npcs::Int, nreasons::Int)
    pcs = Vector{CUpti_PCSamplingPCData}(undef, npcs)
    reasons = Matrix{CUpti_PCSamplingStallReason}(undef, nreasons, npcs)
    GC.@preserve pcs reasons for i in 1:npcs
        ptr = pointer(pcs, i)
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0,
              sizeof(CUpti_PCSamplingPCData))
        ptr.size = sizeof(CUpti_PCSamplingPCData)
        ptr.stallReason = pointer(reasons, (i-1)*nreasons+1)
    end
    data = Ref{CUpti_PCSamplingData}()
    GC.@preserve data begin
        ptr = Base.unsafe_convert(Ptr{CUpti_PCSamplingData}, data)
        ccall(:memset, Ptr{Cvoid}, (Ptr{Cvoid}, Cint, Csize_t), ptr, 0,
              sizeof(CUpti_PCSamplingData))
        ptr.size = sizeof(CUpti_PCSamplingData)
        ptr.collectNumPcs = npcs
        ptr.pPcData = pointer(pcs)
    end
    PCSamplingBuffer(data, pcs, reasons)
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, buf::PCSamplingBuffer) =
    Ptr{Cvoid}(Base.unsafe_convert(Ptr{CUpti_PCSamplingData}, buf.data))

"""
    CUPTI.PCSamples

PC samples collected by [`CUPTI.pc_sample`](@ref):

- `reasons`: the name of every stall reason, by CUPTI's index;
- `pcs`: for every sampled PC, keyed by the CRC of its cubin (as computed by
  [`CUPTI.cubin_crc`](@ref)), its function's name and the offset of the PC in that function,
  the number of samples per stall reason index;
- `total`: the number of samples, including dropped ones and samples in kernels that
  CUPTI doesn't report (e.g. in the driver);
- `dropped`: the number of samples that were lost because CUPTI's buffers were full.
"""
struct PCSamples
    reasons::Dict{UInt32,String}
    pcs::Dict{Tuple{UInt64,String,UInt64},Dict{UInt32,Int}}
    total::Int
    dropped::Int
end

function add_records!(pcs, buf::PCSamplingBuffer)
    data = buf.data[]
    GC.@preserve buf for i in 1:data.totalNumPcs
        ptr = pointer(buf.pcs, i)
        record = unsafe_load(ptr)
        # function names are owned by CUPTI, which may share them across records
        key = (record.cubinCrc, unsafe_string(record.functionName), record.pcOffset)
        counts = get!(pcs, key) do
            Dict{UInt32,Int}()
        end
        for j in 1:record.stallReasonCount
            reason = unsafe_load(record.stallReason, j)
            counts[reason.pcSamplingStallReasonIndex] =
                get(counts, reason.pcSamplingStallReasonIndex, 0) + reason.samples
        end
    end
    return (; total=Int(data.totalSamples), dropped=Int(data.droppedSamples))
end

"""
    CUPTI.pc_sample(f; ctx=context(), period=14) -> CUPTI.PCSamples

Sample the PCs of warps executing kernels in `ctx` while executing `f()`, once every
`2^period` cycles on each SM. The caller should synchronize the context before and after
`f`, as samples are only collected while sampling is active.

!!! warning
    With CUPTI 13.2 and later, loading a module after PC sampling was disabled makes
    enabling PC sampling or the range profiler fail with `CUPTI_ERROR_INVALID_OPERATION`,
    even in other contexts. Call `cuptiFinalize` when CUPTI isn't in use anymore (e.g.,
    after unsubscribing from callbacks) to avoid that.
"""
function pc_sample(f; ctx::CuContext=context(), period::Integer=14)
    5 <= period <= 31 || throw(ArgumentError("The sampling period must be between 5 and 31 (2^5 to 2^31 cycles)."))
    Threads.atomic_cas!(profiler_active, false, true) &&
        error("Another profiling session is already active; PC sampling cannot be used concurrently.")
    try
        check_pc_sampling_support(device(ctx))
        pc_sample(f, ctx, period)
    finally
        profiler_active[] = false
    end
end

function pc_sample(f, ctx::CuContext, period::Integer)
    cuptiPCSamplingEnable(profiler_params(CUpti_PCSamplingEnableParams, :ctx; ctx))
    reasons = try
        stall_reasons(ctx)
    catch
        unchecked_cuptiPCSamplingDisable(profiler_params(CUpti_PCSamplingDisableParams, :ctx; ctx))
        rethrow()
    end

    # the buffer passed during configuration receives the records that are left when
    # disabling sampling. it must differ from the buffer we retrieve data with, or no data
    # is retrieved.
    final = PCSamplingBuffer(1024, length(reasons))
    buffer = PCSamplingBuffer(4096, length(reasons))
    attrs = [
        pc_sampling_attribute(CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_COLLECTION_MODE,
                              UInt32(CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS)),
        pc_sampling_attribute(CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD,
                              UInt32(period)),
        pc_sampling_attribute(CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_ENABLE_START_STOP_CONTROL,
                              UInt32(1)),
        pc_sampling_attribute(CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_DATA_BUFFER,
                              Base.unsafe_convert(Ptr{Cvoid}, final)),
    ]

    pcs = Dict{Tuple{UInt64,String,UInt64},Dict{UInt32,Int}}()
    total = dropped = 0
    GC.@preserve final buffer begin
        try
            set_pc_sampling_attributes(ctx, attrs)

            cuptiPCSamplingStart(profiler_params(CUpti_PCSamplingStartParams, :ctx; ctx))
            try
                f()
            finally
                cuptiPCSamplingStop(profiler_params(CUpti_PCSamplingStopParams, :ctx; ctx))
            end

            # retrieve all records, which may take several calls
            params = profiler_params(CUpti_PCSamplingGetDataParams, :pcSamplingData; ctx,
                                     pcSamplingData=Base.unsafe_convert(Ptr{Cvoid}, buffer))
            while true
                res = unchecked_cuptiPCSamplingGetData(params)
                if res == CUPTI_ERROR_OUT_OF_MEMORY || buffer.data[].hardwareBufferFull != 0
                    error("PC sampling collected more samples than fit in the hardware buffer. Use a longer sampling period.")
                end
                res == CUPTI_SUCCESS || throw_api_error(res)
                counts = add_records!(pcs, buffer)
                total += counts.total
                dropped += counts.dropped
                buffer.data[].remainingNumPcs == 0 && break
            end
        catch
            # after a failed session, disabling can fail too (e.g. CUPTI 13.0 when another
            # process holds the profiling hardware), which shouldn't hide the actual error
            unchecked_cuptiPCSamplingDisable(profiler_params(CUpti_PCSamplingDisableParams, :ctx; ctx))
            rethrow()
        end
        cuptiPCSamplingDisable(profiler_params(CUpti_PCSamplingDisableParams, :ctx; ctx))
        counts = add_records!(pcs, final)
        total += counts.total
        dropped += counts.dropped
    end

    PCSamples(reasons, pcs, total, dropped)
end

"""
    CUPTI.cubin_crc(cubin::Vector) -> UInt64

The CRC that CUPTI uses to identify `cubin` in PC sampling records.
"""
function cubin_crc(cubin::Vector)
    params = GC.@preserve cubin begin
        params = profiler_params(CUpti_GetCubinCrcParams, :cubinCrc;
                                 cubinSize=sizeof(cubin), cubin=pointer(cubin))
        cuptiGetCubinCrc(params)
        params
    end
    params[].cubinCrc
end

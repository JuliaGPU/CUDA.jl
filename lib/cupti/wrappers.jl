function version()
    version_ref = Ref{Cuint}()
    cuptiGetVersion(version_ref)
    VersionNumber(version_ref[])
end


#
# activity API
#

"""
    cfg = ActvitiyConfig(activity_kinds)

    enable!(cfg)
    # do stuff
    disable!(cfg)

    process(cfg) do (ctx, stream, record)
        # inspect record
    end

High-level interface to the CUPTI activity API.
"""
struct ActivityConfig
    activity_kinds::Vector{CUpti_ActivityKind}

    available_buffers::Vector{Vector{UInt8}}
    active_buffers::Vector{Vector{UInt8}}

    results::Vector{Any}
end

function ActivityConfig(activity_kinds)
    # pre-allocate a couple of buffers to avoid allocating while profiling
    available_buffers = [allocate_buffer() for _ in 1:5]

    ActivityConfig(activity_kinds, available_buffers, Vector{UInt8}[], [])
end

function allocate_buffer()
    # "for typical workloads, it's suggested to choose a size between 1 and 10 MB."
    Array{UInt8}(undef, 8 * 1024 * 1024)  # 8 MB
end

const activity_config = Ref{Union{Nothing,ActivityConfig}}(nothing)

function request_buffer(dest_ptr, sz_ptr, max_num_records_ptr)
    # this function is called by CUPTI, but directly from the application, so it should be
    # fine to perform I/O or allocate memory here.

    dest = Base.unsafe_wrap(Array, dest_ptr, 1)
    sz = Base.unsafe_wrap(Array, sz_ptr, 1)
    max_num_records = Base.unsafe_wrap(Array, max_num_records_ptr, 1)

    cfg = activity_config[]
    if cfg !== nothing
        buf = if isempty(cfg.available_buffers)
            allocate_buffer()
        else
            pop!(cfg.available_buffers)
        end
        ptr = pointer(buf)
        push!(cfg.active_buffers, buf)
        sizehint!(cfg.results, length(cfg.active_buffers))

        dest[] = pointer(buf)
        sz[] = sizeof(buf)
        max_num_records[] = 0
    else
        dest[] = C_NULL
        sz[] = 0
        max_num_records[] = 0
    end

    return
end

function complete_buffer(ctx_handle, stream_id, buf_ptr, sz, valid_sz)
    # this function is called by a CUPTI worker thread while our application may be waiting
    # for `cuptiActivityFlushAll` to complete. that means we cannot do I/O here, or we could
    # yield while the application cannot make any progress.
    #
    # we also may not trigger GC, because the main application cannot reach a safepoint.
    # to prevent this, we call `sizehint!` in `request_buffer`.
    # XXX: `sizehint!` isn't a guarantee; use `resize!` and a cursor?

    cfg = activity_config[]
    if cfg !== nothing
        push!(cfg.results, (ctx_handle, stream_id, buf_ptr, sz, valid_sz))
    end

    return
end

function enable!(cfg::ActivityConfig)
    activity_config[] === nothing ||
        error("Only one profiling session can be active at a time.")

    # set-up callbacks
    request_buffer_ptr = @cfunction(request_buffer, Cvoid,
                                    (Ptr{Ptr{UInt8}}, Ptr{Csize_t}, Ptr{Csize_t}))
    complete_buffer_ptr = @cfunction(complete_buffer, Cvoid,
                                     (CUDA.CUcontext, UInt32, Ptr{UInt8}, Csize_t, Csize_t))
    cuptiActivityRegisterCallbacks(request_buffer_ptr, complete_buffer_ptr)

    activity_config[] = cfg

    # enable requested activity kinds
    for activity_kind in cfg.activity_kinds
        cuptiActivityEnable(activity_kind)
    end
end

function disable!(cfg::ActivityConfig)
    if activity_config[] !== cfg
        error("This profiling session is not active.")
    end

    # disable activity kinds
    for activity_kind in cfg.activity_kinds
        cuptiActivityDisable(activity_kind)
    end

    # flush all activity records, even incomplete ones
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)

    activity_config[] = nothing

    return
end

function process(f, cfg::ActivityConfig)
    activity_types = Dict(
        CUPTI_ACTIVITY_KIND_DRIVER              => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_RUNTIME             => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API => CUpti_ActivityAPI,
        CUPTI_ACTIVITY_KIND_NAME                => CUpti_ActivityName,
        CUPTI_ACTIVITY_KIND_MARKER              => CUpti_ActivityMarker2,
        CUPTI_ACTIVITY_KIND_MARKER_DATA         => CUpti_ActivityMarkerData,
    )
    # NOTE: the CUPTI version is unreliable, e.g., both CUDA 11.5 and 11.6 have CUPTI 16,
    #       yet CUpti_ActivityMemset4 is only available in CUDA 11.6.
    cuda_version = CUDA.runtime_version()
    ## kernel activities
    activity_types[CUPTI_ACTIVITY_KIND_KERNEL] =
        if cuda_version >= v"12.0"
            CUpti_ActivityKernel5
        elseif cuda_version >= v"11.8"
            CUpti_ActivityKernel8
        elseif cuda_version >= v"11.6"
            CUpti_ActivityKernel7
        elseif cuda_version >= v"11.2"
            CUpti_ActivityKernel6
        elseif cuda_version >= v"11.1"
            CUpti_ActivityKernel5
        else # v"11.0"
            CUpti_ActivityKernel4
        end
    activity_types[CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL] =
        activity_types[CUPTI_ACTIVITY_KIND_KERNEL]
    ## memcpy activities
    activity_types[CUPTI_ACTIVITY_KIND_MEMCPY] =
        if cuda_version >= v"11.6"
            CUpti_ActivityMemcpy5
        elseif cuda_version >= v"11.1"
            CUpti_ActivityMemcpy4
        else # v"11.0"
            CUpti_ActivityMemcpy3
        end
    activity_types[CUPTI_ACTIVITY_KIND_MEMSET] =
        if cuda_version >= v"11.6"
            CUpti_ActivityMemset4
        elseif cuda_version >= v"11.1"
            CUpti_ActivityMemset3
        else # v"11.0"
            CUpti_ActivityMemset2
        end
    activity_types[CUPTI_ACTIVITY_KIND_MEMORY2] =
        if cuda_version >= v"11.6"
            CUpti_ActivityMemory3
        elseif cuda_version >= v"11.2"
            CUpti_ActivityMemory2
        else # v"9.0"
            CUpti_ActivityMemory
        end

    # extract typed activity records
    for (ctx_handle, stream_id, buf_ptr, sz, valid_sz) in cfg.results
        ctx = CUDA._CuContext(ctx_handle)
        # XXX: can we reconstruct the stream from the stream ID?

        record_ptr = Ref{Ptr{CUpti_Activity}}(C_NULL)
        while true
            try
                cuptiActivityGetNextRecord(buf_ptr, valid_sz, record_ptr)
                record = unsafe_load(record_ptr[])

                if haskey(activity_types, record.kind)
                    typ = activity_types[record.kind]
                    typed_ptr = convert(Ptr{typ}, record_ptr[])
                    typed_record = unsafe_load(typed_ptr)
                    f(ctx, stream_id, typed_record)
                else
                    @warn """Unsupported activity kind: $(record.kind).
                             Please file an issue, or extend the implementation of `CUPTI.process` to handle this activity kind."""
                end
            catch err
                if isa(err, CUPTIError) && err.code == CUPTI_ERROR_MAX_LIMIT_REACHED
                    break
                end
                rethrow()
            end
        end
    end
end

function Base.string(memory_kind::CUpti_ActivityMemoryKind)
    if memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN
        "unknown"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE
        "pageable"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_PINNED
        "pinned"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE
        "device"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_ARRAY
        "array"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED
        "managed"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC
        "device static"
    elseif memory_kind == CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC
        "managed static"
    else
        "unknown"
    end
end

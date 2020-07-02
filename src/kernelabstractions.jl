# KernelAbstractions.jl interface

module CUDAKA
    import KernelAbstractions
    const KA = KernelAbstractions
    import ..CUDA

    struct CUDADevice <: KA.GPU end
    export CUDADevice

    const FREE_STREAMS = CUDA.CuStream[]
    const STREAMS = CUDA.CuStream[]
    const STREAM_GC_THRESHOLD = Ref{Int}(16)

    ## 
    # Stream GC
    ##

    # Simplistic stream gc design in which when we have a total number
    # of streams bigger than a threshold, we start scanning the streams
    # and add them back to the freelist if all work on them has completed.
    # Alternative designs:
    # - Enqueue a host function on the stream that adds the stream back to the freelist
    # - Attach a finalizer to events that adds the stream back to the freelist
    # Possible improvements
    # - Add a background task that occasionally scans all streams
    # - Add a hysterisis by checking a "since last scanned" timestamp
    # - Add locking
    function next_stream()
        if !isempty(FREE_STREAMS)
            return pop!(FREE_STREAMS)
        end

        if length(STREAMS) > STREAM_GC_THRESHOLD[]
            for stream in STREAMS
                if CUDA.query(stream)
                    push!(FREE_STREAMS, stream)
                end
            end
        end

        if !isempty(FREE_STREAMS)
            return pop!(FREE_STREAMS)
        end
        stream = CUDA.CuStream(flags = CUDA.STREAM_NON_BLOCKING)
        push!(STREAMS, stream)
        return stream
    end

    ##
    # KernelAbstractions event
    ##
    import KernelAbstractions: Event, NoneEvent, MultiEvent, CPUEvent
    import KernelAbstractions: wait, unsafe_wait

    struct CudaEvent <: Event
        event::CUDA.CuEvent
    end

    failed(::CudaEvent) = false
    isdone(ev::CudaEvent) = CUDA.query(ev.event)

    """
        Event(::CUDADevice)

    Place a KernelAbstractions event onto the CuDefaultStream,
    to synchronize KernelAbstractions code with CUDA.jl code.
    """
    function Event(::CUDADevice)
        stream = CUDA.CuDefaultStream()
        event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
        CUDA.record(event, stream)
        CudaEvent(event)
    end

    wait(ev::CudaEvent, progress=yield) = wait(KA.CPU(), ev, progress)

    function wait(::KA.CPU, ev::CudaEvent, progress=yield)
        if progress === nothing
            CUDA.synchronize(ev.event)
        else
            while !isdone(ev)
                progress()
            end
        end
    end

    # Use this to synchronize between computation using the CuDefaultStream
    wait(::CUDADevice, ev::CudaEvent, progress=nothing, stream=CUDA.CuDefaultStream()) = CUDA.wait(ev.event, stream)
    wait(::CUDADevice, ev::NoneEvent, progress=nothing, stream=nothing) = nothing

    function wait(::CUDADevice, ev::MultiEvent, progress=nothing, stream=CUDA.CuDefaultStream())
        dependencies = collect(ev.events)
        cudadeps  = filter(d->d isa CudaEvent,    dependencies)
        otherdeps = filter(d->!(d isa CudaEvent), dependencies)
        for event in cudadeps
            CUDA.wait(event.event, stream)
        end
        for event in otherdeps
            wait(CUDADevice(), event, progress, stream)
        end
    end

    # include("cusynchronization.jl")
    # import .CuSynchronization: unsafe_volatile_load, unsafe_volatile_store!

    function wait(::CUDADevice, ev::CPUEvent, progress=nothing, stream=nothing)
        error("""
        Waiting on the GPU for an CPU event to finish is currently not supported.
        We have encountered deadlocks arising, due to interactions with the CUDA
        driver. If you are certain that you are deadlock free, you can use `unsafe_wait`
        instead.
        """)
    end

    function unsafe_wait(::CUDADevice, ev::CPUEvent, progress=nothing, stream=CUDA.CuDefaultStream())
        error("TODO: Not implemented yet")
    end

    ###
    # pinning
    # - IdDict does not free the memory
    # - WeakRef dict does not unique the key by objectid
    # TODO: Implement in a re-usable fashion.
    ###
    const __pinned_memory = Dict{UInt64, WeakRef}()

    function __pin!(a)
        # use pointer instead of objectid?
        oid = objectid(a)
        if haskey(__pinned_memory, oid) && __pinned_memory[oid].value !== nothing
            return nothing
        end
        ad = CUDA.Mem.register(CUDA.Mem.Host, pointer(a), sizeof(a))
        finalizer(_ -> CUDA.Mem.unregister(ad), a)
        __pinned_memory[oid] = WeakRef(a)
        return nothing
    end

    ###
    # async_copy
    ###
    function async_copy!(::CUDADevice, A, B; dependencies=nothing, progress=yield)
        A isa Array && __pin!(A)
        B isa Array && __pin!(B)

        stream = next_stream()
        wait(CUDADevice(), MultiEvent(dependencies), progress, stream)
        event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
        GC.@preserve A B begin
            destptr = pointer(A)
            srcptr  = pointer(B)
            N       = length(A)
            unsafe_copyto!(destptr, srcptr, N, async=true, stream=stream)
        end

        CUDA.record(event, stream)
        return CudaEvent(event)
    end

    ###
    # Kernel launch
    ###
    import KernelAbstractions: Kernel
    import KernelAbstractions: Cassette

    function (obj::Kernel{CUDADevice})(args...; ndrange=nothing, dependencies=nothing, workgroupsize=nothing, progress=yield)
        if ndrange isa Integer
            ndrange = (ndrange,)
        end
        if workgroupsize isa Integer
            workgroupsize = (workgroupsize, )
        end

        if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
            # TODO: allow for NDRange{1, DynamicSize, DynamicSize}(nothing, nothing)
            #       and actually use CUDA autotuning
            workgroupsize = (256,)
        end
        # If the kernel is statically sized we can tell the compiler about that
        if KA.workgroupsize(obj) <: KA.StaticSize
            maxthreads = prod(get(KA.workgroupsize(obj)))
        else
            maxthreads = nothing
        end

        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)

        nblocks = length(KA.blocks(iterspace))
        threads = length(KA.workitems(iterspace))

        if nblocks == 0
            return MultiEvent(dependencies)
        end

        stream = next_stream()
        wait(CUDADevice(), MultiEvent(dependencies), progress, stream)

        ctx = mkcontext(obj, ndrange, iterspace)
        # Launch kernel
        event = CUDA.CuEvent(CUDA.EVENT_DISABLE_TIMING)
        CUDA.@cuda(threads=threads, blocks=nblocks, stream=stream,
                   name=String(nameof(obj.f)), maxthreads=maxthreads,
                   Cassette.overdub(ctx, obj.f, args...))

        CUDA.record(event, stream)
        return CudaEvent(event)
    end

    ###
    # Backend functionality
    ###

    Cassette.@context CUDACtx

    function mkcontext(kernel::Kernel{CUDADevice}, _ndrange, iterspace)
        metadata = CompilerMetadata{KA.ndrange(kernel), KA.DynamicCheck}(_ndrange, iterspace)
        Cassette.disablehooks(CUDACtx(pass = KA.CompilerPass, metadata=metadata))
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Local_Linear))
        return CUDA.threadIdx().x
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Group_Linear))
        return CUDA.blockIdx().x
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Global_Linear))
        I =  @inbounds KA.expand(KA.__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
        # TODO: This is unfortunate, can we get the linear index cheaper
        @inbounds LinearIndices(KA.__ndrange(ctx.metadata))[I]
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Local_Cartesian))
        @inbounds KA.workitems(KA.__iterspace(ctx.metadata))[CUDA.threadIdx().x]
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Group_Cartesian))
        @inbounds KA.blocks(KA.__iterspace(ctx.metadata))[CUDA.blockIdx().x]
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__index_Global_Cartesian))
        return @inbounds KA.expand(KA.__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__validindex))
        if KA.__dynamic_checkbounds(ctx.metadata)
            I = @inbounds KA.expand(KA.__iterspace(ctx.metadata), CUDA.blockIdx().x, CUDA.threadIdx().x)
            return I in KA.__ndrange(ctx.metadata)
        else
            return true
        end
    end

    KA.generate_overdubs(@__MODULE__, CUDACtx)

    ###
    # CUDA specific method rewrites
    ###

    @inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Float64) = CUDA.pow(x, y)
    @inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Float32) = CUDA.pow(x, y)
    @inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float64, y::Int32)   = CUDA.pow(x, y)
    @inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Float32, y::Int32)   = CUDA.pow(x, y)
    @inline Cassette.overdub(::CUDACtx, ::typeof(^), x::Union{Float32, Float64}, y::Int64) = CUDA.pow(x, y)

    # libdevice.jl
    const cudafuns = (:cos, :cospi, :sin, :sinpi, :tan,
              :acos, :asin, :atan,
              :cosh, :sinh, :tanh,
              :acosh, :asinh, :atanh,
              :log, :log10, :log1p, :log2,
              :exp, :exp2, :exp10, :expm1, :ldexp,
              # :isfinite, :isinf, :isnan, :signbit,
              :abs,
              :sqrt, :cbrt,
              :ceil, :floor,)
    for f in cudafuns
        @eval function Cassette.overdub(ctx::CUDACtx, ::typeof(Base.$f), x::Union{Float32, Float64})
            @Base._inline_meta
            return CUDA.$f(x)
        end
    end

    @inline Cassette.overdub(::CUDACtx, ::typeof(sincos), x::Union{Float32, Float64}) = (CUDA.sin(x), CUDA.cos(x))
    @inline Cassette.overdub(::CUDACtx, ::typeof(exp), x::Union{ComplexF32, ComplexF64}) = CUDA.exp(x)

    import SpecialFunctions
    @inline Cassette.overdub(::CUDACtx, ::typeof(SpecialFunctions.gamma), x::Union{Float32, Float64}) = CUDA.tgamma(x)

    ###
    # GPU implementation of shared memory
    ###
    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.SharedMemory), ::Type{T}, ::Val{Dims}, ::Val{Id}) where {T, Dims, Id}
        ptr = CUDA._shmem(Val(Id), T, Val(prod(Dims)))
        CUDA.CuDeviceArray(Dims, ptr)
    end

    ###
    # GPU implementation of scratch memory
    # - private memory for each workitem
    ###
    using StaticArrays
    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.Scratchpad), ::Type{T}, ::Val{Dims}) where {T, Dims}
        MArray{__size(Dims), T}(undef)
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__synchronize))
        CUDA.sync_threads()
    end

    @inline function Cassette.overdub(ctx::CUDACtx, ::typeof(KA.__print), args...)
        CUDA._cuprint(args...)
    end

    ###
    # GPU implementation of `@Const`
    ##
    import Adapt 
    import KernelAbstractions: ConstAdaptor
    Adapt.adapt_storage(::ConstAdaptor, a::CUDA.CuDeviceArray) = CUDA.Const(a)
end
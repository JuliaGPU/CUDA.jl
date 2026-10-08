# CUDA graphs
#
# a graph records a sequence of GPU operations, so that it can be launched as a whole, with
# much less CPU overhead than launching each operation separately. graphs are created by
# capturing the operations that are executed on a stream, and can be instantiated into
# executable graphs that can be launched many times.
#
# the main challenge is integrating with CUDA.jl's memory management: graphs use memory
# whenever they are launched, long after the arrays that the captured operations used may
# have been freed. so graphs lease all memory that is used by captured operations, which
# postpones releasing it until the graph is gone (see `lease!`). executable graphs also take
# ownership of that memory when they are launched, just like kernel launches do, so that the
# usual synchronization between tasks keeps working, and so that releasing memory is ordered
# after the last launch of the graph.

export CuGraph, CuGraphExec, CuGraphNode, capture, capture!, instantiate, launch, update,
       update!, upload, @captured
@public nodes


## capture state

# bookkeeping for a capture in progress, identified by its capture ID. this is needed to
# find the capture that a stream is part of, which may be a different stream than the one the
# capture was started on (when capturing operations on multiple streams).
mutable struct CaptureState
    # the stream the capture was started on
    const stream::CuStream
    # memory used by the captured operations, which is leased for every use recorded here
    const memory::Base.IdSet{Managed}
    # the ID of the capture, once it has been registered
    id::Union{Nothing,UInt64}
    const lock::Threads.SpinLock
end
CaptureState(stream::CuStream) =
    CaptureState(stream, Base.IdSet{Managed}(), nothing, Threads.SpinLock())

function register!(capture::CaptureState)
    id = something(capture_id(capture.stream))
    stream = capture.stream
    @lock captures_lock begin
        captures[id] = capture
        stream.ctx === nothing || (captures_by_stream[stream.handle] = capture)
    end
    capture.id = id
    return
end

function unregister!(capture::CaptureState)
    id = capture.id
    id === nothing && return
    stream = capture.stream
    @lock captures_lock begin
        delete!(captures, id)
        stream.ctx === nothing || delete!(captures_by_stream, stream.handle)
    end
    capture.id = nothing
    return
end

# captures in progress, by capture ID, and by the stream they were started on (which avoids
# having to look up the capture ID for most operations). the special streams, which don't
# have a context, have handles that aren't unique, so are only looked up by capture ID.
const captures = Dict{UInt64,CaptureState}()
const captures_by_stream = Dict{CUstream,CaptureState}()
const captures_lock = Threads.SpinLock()

function capture_id(stream::CuStream)
    status = Ref{CUstreamCaptureStatus}()
    id = Ref{UInt64}()
    cuStreamGetCaptureInfo(stream, status, id)
    status[] == STREAM_CAPTURE_STATUS_NONE && return nothing
    return id[]
end

# the capture by `capture` or `capture!` that `stream` is part of, if any. must only be
# called for streams that are being captured.
function current_capture(stream::CuStream)
    if stream.ctx !== nothing
        capture = @lock captures_lock get(captures_by_stream, stream.handle, nothing)
        capture === nothing || return capture
    end
    id = capture_id(stream)
    id === nothing && return nothing
    @lock captures_lock get(captures, id, nothing)
end

# record that the capture uses memory, leasing it for the graph
function record!(capture::CaptureState, managed::Managed)
    @lock capture.lock begin
        managed in capture.memory && return
        push!(capture.memory, lease!(managed))
    end
    return
end

## memory allocation

# memory allocated in stream order while capturing a graph would become owned by that graph,
# only valid after launching it, and only until launching it again (or destroying it). that
# is not compatible with arrays whose lifetime is managed by the GC, so allocate memory on a
# separate stream instead, and keep it alive for as long as the graph may use it.
@noinline function capture_alloc(::Type{B}, sz, state) where {B<:AbstractMemory}
    alloc_stream = allocation_stream(state.context)
    time = Base.@elapsed begin
        # (allocating on another stream isn't allowed while capturing)
        mem = relaxed_capture_mode() do
            _pool_alloc(B, sz, (; state..., stream=alloc_stream))
        end

        if B != DeviceMemory
            # other memory can be accessed by the CPU, so wait for the allocation right away
            res = relaxed_capture_mode(() -> unchecked_synchronize(alloc_stream))
            res == SUCCESS || throw_api_error(res)
        end
    end

    Base.@atomic alloc_stats.alloc_count += 1
    Base.@atomic alloc_stats.alloc_bytes += sz
    Base.@atomic alloc_stats.total_time += time

    # the memory is owned by the allocation stream, and dirty, so that whoever uses it next
    # (launching the graph, or another task) waits for the allocation, and for any
    # initialization that is performed on that stream before the memory is published. the
    # captured operations themselves only record their use (see `take_ownership!`).
    return Managed(mem; stream=alloc_stream, owned_allocation=true)
end

# a non-blocking stream per context, used to allocate memory while capturing graphs
const allocation_streams = Dict{CuContext,CuStream}()
const allocation_streams_lock = Threads.SpinLock()
function allocation_stream(ctx::CuContext)
    @lock allocation_streams_lock get!(allocation_streams, ctx) do
        context!(() -> CuStream(; flags=STREAM_NON_BLOCKING), ctx)
    end
end

# initialize memory that was allocated during capture with a (small) value, without capturing
# the operation. this is ordered after the allocation on the stream that owns the memory, so
# users of the memory wait for it too. memsets are used, as copying from unpinned memory
# would wait for the stream, and as such for any work the allocation depends on.
function initialize_during_capture(managed::Managed, value::T) where {T}
    ref = Ref(value)
    nbytes = aligned_sizeof(T)
    ptr = convert(CuPtr{UInt8}, managed.mem)
    GC.@preserve ref relaxed_capture_mode() do
        stream = allocation_stream(managed.mem.ctx)
        src = Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, ref))
        for i in 0:4:nbytes-4
            cuMemsetD32Async(ptr + i, unsafe_load(Ptr{UInt32}(src + i)), 1, stream)
        end
        for i in (nbytes - nbytes % 4):nbytes-1
            cuMemsetD8Async(ptr + i, unsafe_load(src + i), 1, stream)
        end
    end
    return
end

# host memory read by a captured copy needs to stay valid for as long as the graph may be
# launched. copy it to pinned memory, which the graph keeps alive. (this gives captured
# copies snapshot semantics, which is also what library calls taking scalar arguments by
# reference need.)
function stage_host_memory(src::Ptr{T}, nbytes::Integer, stream::CuStream) where {T}
    capture = current_capture(stream)
    capture === nothing &&
        throw(CaptureError("cannot copy from host memory while capturing a graph using the driver API"))
    managed = capture_alloc(HostMemory, nbytes, active_state())
    staging = convert(Ptr{UInt8}, managed.mem)
    Base.unsafe_copyto!(staging, convert(Ptr{UInt8}, src), nbytes)
    record!(capture, managed)
    pool_free(managed)  # (only when the graph is gone)
    return convert(Ptr{T}, staging)
end

## graphs

"""
    CuGraph()

A graph of GPU operations. Graphs are typically created by recording operations with
[`capture`](@ref), but you can also create an empty graph and add operations to it using
[`capture!`](@ref). To execute a graph, [`instantiate`](@ref) it.

A graph keeps the memory that its operations use alive for as long as it exists.
"""
mutable struct CuGraph
    const handle::CUgraph
    const ctx::CuContext

    # memory used by the graph's operations, leased for as long as the graph exists
    const memory::Base.IdSet{Managed}

    # the driver requires graphs to be used by one thread at a time
    const lock::ReentrantLock

    function CuGraph(handle::CUgraph, ctx::CuContext=context())
        obj = new(handle, ctx, Base.IdSet{Managed}(), ReentrantLock())
        resource_finalizer(obj)
        return obj
    end
end

function CuGraph()
    handle_ref = Ref{CUgraph}()
    cuGraphCreate(handle_ref, 0)
    return CuGraph(handle_ref[])
end

Base.unsafe_convert(::Type{CUgraph}, graph::CuGraph) = graph.handle

function release_now(graph::CuGraph; destroy=cuGraphDestroy)
    context!(graph.ctx) do
        destroy(graph)
    end
    # (not reached when destroying failed: the resource coordinator then retains the object,
    # so its memory needs to stay leased)
    foreach(unlease!, graph.memory)
    return
end

"""
    nodes(graph::CuGraph)

Return the nodes of a graph.
"""
function nodes(graph::CuGraph)
    handles = @lock graph.lock begin
        count = Ref{Csize_t}(0)
        cuGraphGetNodes(graph, C_NULL, count)
        handles = Vector{CUgraphNode}(undef, count[])
        # the driver rejects requests for zero nodes
        isempty(handles) || cuGraphGetNodes(graph, handles, count)
        resize!(handles, count[])
    end
    return CuGraphNode[CuGraphNode(handle, graph) for handle in handles]
end

Base.length(graph::CuGraph) = length(nodes(graph))

function Base.show(io::IO, graph::CuGraph)
    print(io, "CuGraph(")
    @printf(io, "%p", graph.handle)
    print(io, ")")
end

function Base.show(io::IO, ::MIME"text/plain", graph::CuGraph)
    types = map(nodetype, nodes(graph))
    print(io, "CuGraph with ", length(types), length(types) == 1 ? " node" : " nodes")
    counts = Dict{CUgraphNodeType,Int}()
    for type in types
        counts[type] = get(counts, type, 0) + 1
    end
    isempty(counts) && return
    print(io, ": ")
    join(io, ["$n $(lowercase(string(type)[length("CU_GRAPH_NODE_TYPE_")+1:end]))"
              for (type, n) in sort!(collect(counts); by=last, rev=true)], ", ")
end

# render as Graphviz, e.g., for use with the `dot` command or a notebook
function Base.show(io::IO, ::MIME"text/vnd.graphviz", graph::CuGraph)
    mktemp() do path, file
        close(file)
        @lock graph.lock cuGraphDebugDotPrint(graph, path, 0)
        write(io, read(path))
    end
    return
end

# memory allocations that a graph makes without freeing them. these remain allocated after
# launching the graph, and need to be freed before the graph can be launched again. we don't
# allocate memory like that (see `capture_alloc`), but libraries may.
function unfreed_allocations(graph::CuGraph)
    @lock graph.lock unfreed_allocations_locked(graph)
end
function unfreed_allocations_locked(graph::CuGraph)
    allocated = CuPtr{Cvoid}[]
    freed = CuPtr{Cvoid}[]
    for node in nodes(graph)
        type = nodetype(node)
        if type == CU_GRAPH_NODE_TYPE_MEM_ALLOC
            params = Ref{CUDA_MEM_ALLOC_NODE_PARAMS}()
            cuGraphMemAllocNodeGetParams(node, params)
            push!(allocated, reinterpret(CuPtr{Cvoid}, params[].dptr))
        elseif type == CU_GRAPH_NODE_TYPE_MEM_FREE
            ptr = Ref{CUdeviceptr}()
            cuGraphMemFreeNodeGetParams(node, ptr)
            push!(freed, reinterpret(CuPtr{Cvoid}, ptr[]))
        end
    end
    return setdiff(allocated, freed)
end


## graph nodes

"""
    CuGraphNode

A node in a [`CuGraph`](@ref), representing an operation. Nodes are owned by their graph,
and are returned by [`capture!`](@ref) to express dependencies between operations.
"""
struct CuGraphNode
    handle::CUgraphNode
    graph::CuGraph
end

Base.unsafe_convert(::Type{CUgraphNode}, node::CuGraphNode) = node.handle

@enum_without_prefix visibility=:public CUgraphNodeType CU_

function nodetype(node::CuGraphNode)
    type = Ref{CUgraphNodeType}()
    @lock node.graph.lock cuGraphNodeGetType(node, type)
    return type[]
end

function Base.show(io::IO, node::CuGraphNode)
    print(io, "CuGraphNode(")
    @printf(io, "%p", node.handle)
    print(io, ")")
end


## capture

# operations are captured on a dedicated stream, instead of on the task's stream. that keeps
# the task's stream usable by other tasks waiting for work that was submitted to it before
# the capture (e.g., to use an array that the capturing task produced). capture streams are
# non-blocking, so that they don't synchronize with the legacy default stream, and are only
# ever used by one capture at a time. nothing executes on them, so they can be reused right
# after the capture has ended.
const capture_streams = Dict{Tuple{CuContext,Cint},Vector{CuStream}}()
# streams that are in use by a capture, including ones passed to `capture` explicitly
const busy_capture_streams = Set{CuStream}()
const capture_streams_lock = Threads.SpinLock()

function checkout_capture_stream(ctx::CuContext, priority::Cint)
    stream = @lock capture_streams_lock begin
        pool = get(capture_streams, (ctx, priority), CuStream[])
        while !isempty(pool)
            stream = pop!(pool)
            # (a stream may have been passed to `capture` explicitly since it was released)
            if !(stream in busy_capture_streams)
                push!(busy_capture_streams, stream)
                return stream
            end
        end
    end
    stream = context!(ctx) do
        priority == 0 ? CuStream(; flags=STREAM_NON_BLOCKING) :
                        CuStream(; flags=STREAM_NON_BLOCKING, priority)
    end
    @lock capture_streams_lock push!(busy_capture_streams, stream)
    return stream
end

function release_capture_stream(stream::CuStream, priority::Cint)
    @lock capture_streams_lock begin
        delete!(busy_capture_streams, stream)
        # a capture that couldn't be ended leaves its stream capturing
        if isvalid(stream) && !is_capturing(stream)
            push!(get!(Vector{CuStream}, capture_streams,
                       (something(stream.ctx), priority)), stream)
        end
    end
    return
end

# make `stream` the stream of the current task for the duration of `f`
function with_task_stream(f, stream::CuStream)
    state = task_local_state!()
    devidx = deviceid(state.device)+1
    old = state.streams[devidx]
    state.streams[devidx] = stream
    try
        # (so that the task's stream isn't changed while capturing, see `priority!`)
        task_local_storage(f, :CUDA_capture_stream, stream)
    finally
        state.streams[devidx] = old
    end
end

# capture the operations performed by `f` on a dedicated stream, either into a new graph,
# or into an existing one (depending on the nodes in `deps`). returns the graph, or
# `nothing` if capturing failed and `throw_error` is false, along with the nodes that
# operations depending on the captured ones should depend on.
function capture_stream(f, graph::Union{Nothing,CuGraph}, deps::Vector{CUgraphNode};
                        mode::CUstreamCaptureMode, throw_error::Bool,
                        stream::Union{Nothing,CuStream}=nothing)
    ctx = context()
    if stream === nothing
        prio = Cint(priority())
        stream = checkout_capture_stream(ctx, prio)
        try
            capture_on(f, stream, ctx, graph, deps; mode, throw_error)
        finally
            release_capture_stream(stream, prio)
        end
    else
        reserve_capture_stream(stream, ctx)
        try
            capture_on(f, stream, ctx, graph, deps; mode, throw_error)
        finally
            @lock capture_streams_lock delete!(busy_capture_streams, stream)
        end
    end
end

function reserve_capture_stream(stream::CuStream, ctx::CuContext)
    haskey(task_local_storage(), :CUDA_capture_stream) &&
        throw(ArgumentError("Cannot capture on a specific stream while already capturing"))
    stream.ctx == ctx ||
        throw(ArgumentError("Can only capture on a stream of the current context"))
    @lock capture_streams_lock begin
        (stream in busy_capture_streams || is_capturing(stream)) &&
            throw(ArgumentError("Cannot capture on a stream that's already being captured"))
        # (only once we know the stream isn't being captured, as some drivers don't support
        #  querying its flags then, and invalidate the capture instead)
        stream_flags(stream) == STREAM_NON_BLOCKING ||
            throw(ArgumentError("Can only capture on a stream created with `flags=STREAM_NON_BLOCKING`"))
        push!(busy_capture_streams, stream)
    end
    return
end

function capture_on(f, stream::CuStream, ctx::CuContext, graph::Union{Nothing,CuGraph},
                    deps::Vector{CUgraphNode}; mode::CUstreamCaptureMode, throw_error::Bool)
    capture = CaptureState(stream)
    handle = Ref{CUgraph}(C_NULL)

    # captures in the stricter modes have to be ended on the thread they were started on,
    # and only protect that thread. relaxed captures can migrate between threads, as tasks
    # do when they yield.
    task = current_task()
    sticky = task.sticky
    mode == STREAM_CAPTURE_MODE_RELAXED || (task.sticky = true)
    try
        # releasing resources can involve operations that aren't allowed while capturing,
        # so wait for releases in progress to finish, and prevent new ones from starting
        begin_capture()
        try
            if graph === nothing
                cuStreamBeginCapture_v2(stream, mode)
            else
                cuStreamBeginCaptureToGraph(stream, graph, deps, C_NULL, length(deps), mode)
            end

            # from here on, the capture needs to be ended
            try
                register!(capture)
                with_task_stream(f, stream)
            catch err
                try
                    end_capture(capture, handle)
                catch
                    # report the original error
                end
                discard_capture(graph, capture, handle[])
                if !throw_error && unsupported_during_capture(err)
                    return nothing, CUgraphNode[]
                end
                rethrow()
            end
            frontier, res = try
                end_capture(capture, handle)
            catch
                discard_capture(graph, capture, handle[])
                rethrow()
            end
            if res != SUCCESS
                discard_capture(graph, capture, handle[])
                if !throw_error && res == ERROR_STREAM_CAPTURE_INVALIDATED
                    return nothing, CUgraphNode[]
                end
                throw_api_error(res)
            end

            if graph === nothing
                graph = CuGraph(handle[], ctx)
            end
            adopt!(graph, capture)
            return graph, frontier
        finally
            end_capture()
        end
    finally
        task.sticky = sticky

        # release resources that were retired while capturing
        drain_retired()
    end
end

# end a capture, returning the dependencies of the last captured operations, and the result
function end_capture(capture::CaptureState, handle::Ref{CUgraph})
    stream = capture.stream
    local res
    frontier = try
        capture_frontier(stream)
    finally
        res = unchecked_cuStreamEndCapture(stream, handle)
        unregister!(capture)
    end
    return frontier, res
end

function discard_capture(graph::Union{Nothing,CuGraph}, capture::CaptureState,
                         handle::CUgraph)
    if graph === nothing
        # a failed capture may still have produced a graph, which we don't need
        handle == C_NULL || cuGraphDestroy(handle)
        foreach(unlease!, capture.memory)
    else
        # the existing graph may contain some of the captured operations
        adopt!(graph, capture)
    end
    return
end

# the graph now uses the memory that was used by the captured operations
function adopt!(graph::CuGraph, capture::CaptureState)
    @lock graph.lock begin
        for managed in capture.memory
            if managed in graph.memory
                unlease!(managed)
            else
                push!(graph.memory, managed)
            end
        end
    end
    return
end

# whether an exception was caused by an operation that isn't supported during capture
unsupported_during_capture(err) =
    err isa CaptureError ||
    (err isa CuError && err.code in (ERROR_STREAM_CAPTURE_UNSUPPORTED,
                                     ERROR_STREAM_CAPTURE_INVALIDATED,
                                     ERROR_STREAM_CAPTURE_IMPLICIT,
                                     ERROR_CAPTURED_EVENT))

# the nodes that the next captured operation on `stream` would depend on
function capture_frontier(stream::CuStream)
    driver_version() >= v"12.3" || return CUgraphNode[]
    status = Ref{CUstreamCaptureStatus}()
    deps = Ref{Ptr{CUgraphNode}}()
    count = Ref{Csize_t}(0)
    res = unchecked_cuStreamGetCaptureInfo_v3(stream, status, C_NULL, C_NULL, deps, C_NULL,
                                              count)
    (res == SUCCESS && status[] == STREAM_CAPTURE_STATUS_ACTIVE) || return CUgraphNode[]
    return copy(unsafe_wrap(Array, deps[], count[]))
end

"""
    capture(f; [stream], mode=STREAM_CAPTURE_MODE_RELAXED, throw_error=true)::CuGraph

Capture the GPU operations that `f` performs into a graph, without executing them. The
operations are captured on a dedicated stream, which is the task's stream while capturing,
so other tasks can keep using the regular stream of the task. Instantiate the graph to
execute it, possibly many times:

```julia
graph = capture() do
    y .= a .* x .+ y
end
exec = instantiate(graph)
for i in 1:100
    exec()
end
```

Only the GPU operations are captured: launching the graph doesn't run `f`. The operations
are captured with the arguments they were called with, so launching the graph always uses
the same arrays and scalars. The contents of those arrays may change, though, so to execute
the graph with different inputs, copy them into the arrays used during capture.

Memory that is used by captured operations is kept alive by the graph. Arrays that are
allocated during capture are allocated once, outside of the graph, and launching the graph
overwrites their contents, so allocating operations like `y = a .* x` can be captured too.
Data copied from the CPU (e.g., when using `CuRef` scalars) is copied when capturing.

Not all operations can be captured. Anything that waits for the GPU, like copying memory
back to the CPU, or that creates library handles, results in a [`CaptureError`](@ref) or a
`CuError`. It's typically a good idea to execute `f` once before capturing it, so that
kernels are compiled and libraries are initialized. When `throw_error` is false, failures
due to unsupported operations are not reported, and `nothing` is returned instead.

To capture on a specific stream instead, pass it as the `stream` keyword argument. That
needs to be a stream that was created with `flags=STREAM_NON_BLOCKING` in the current
context, and that isn't used by anything else while capturing. Other tasks cannot wait for
work that was submitted to that stream before the capture until the capture has ended.

CUDA.jl checks the operations it performs itself, like waiting for the GPU, but by default
doesn't ask the driver to prohibit other operations that are potentially unsafe during
capture. The driver performs those checks per thread rather than per task, so they would
also reject operations by unrelated tasks that happen to run on the capturing thread while
the capturing task is waiting. To have the driver perform these checks anyway, e.g., to
debug a library that doesn't support capture, use `mode=STREAM_CAPTURE_MODE_THREAD_LOCAL`
(which checks the capturing thread, and keeps the task on that thread), or
`mode=STREAM_CAPTURE_MODE_GLOBAL` (which also checks other threads).

See also: [`instantiate`](@ref), [`capture!`](@ref).
"""
function capture(f::Function; stream::Union{Nothing,CuStream}=nothing,
                 mode::CUstreamCaptureMode=STREAM_CAPTURE_MODE_RELAXED,
                 flags::Union{Nothing,CUstreamCaptureMode}=nothing, throw_error::Bool=true)
    if flags !== nothing
        Base.depwarn("The `flags` keyword argument to `capture` has been renamed to `mode`.",
                     :capture)
        mode = flags
    end
    graph, _ = capture_stream(f, nothing, CUgraphNode[]; mode, throw_error, stream)
    return graph
end

"""
    capture!(f, graph::CuGraph; after=CuGraphNode[], [stream], mode=STREAM_CAPTURE_MODE_RELAXED)

Capture the GPU operations that `f` performs, like [`capture`](@ref), but add them to an
existing graph. The captured operations depend on the nodes in `after`, and are independent
of the other operations in the graph. This makes it possible to construct graphs with
operations that can execute concurrently, without having to capture multiple streams:

```julia
graph = CuGraph()
a = capture!(graph) do
    x .= sin.(x)
end
b = capture!(graph) do
    y .= cos.(y)
end
capture!(graph; after=[a; b]) do
    z .= x .+ y
end
```

Returns the nodes that operations depending on the captured operations should depend on.
If capturing fails, the graph may contain some of the captured operations, and should not
be used anymore.

This functionality requires CUDA 12.3 or higher.
"""
function capture!(f::Function, graph::CuGraph; after::AbstractVector{CuGraphNode}=CuGraphNode[],
                  stream::Union{Nothing,CuStream}=nothing,
                  mode::CUstreamCaptureMode=STREAM_CAPTURE_MODE_RELAXED)
    driver_version() >= v"12.3" ||
        error("Capturing into an existing graph requires CUDA 12.3 or higher")
    all(node -> node.graph === graph, after) ||
        throw(ArgumentError("Dependencies need to be nodes of the same graph"))
    deps = CUgraphNode[node.handle for node in after]
    frontier = @lock graph.lock context!(graph.ctx) do
        _, frontier = capture_stream(f, graph, deps; mode, throw_error=true, stream)
        frontier
    end
    return CuGraphNode[CuGraphNode(node, graph) for node in frontier]
end


## executable graphs

"""
    CuGraphExec

An executable graph, created by instantiating a [`CuGraph`](@ref), which can be launched
many times. It keeps the memory used by the graph's operations alive for as long as it
exists.

See also: [`instantiate`](@ref), [`launch`](@ref), [`update`](@ref).
"""
mutable struct CuGraphExec
    const handle::CUgraphExec
    const ctx::CuContext
    const lock::ReentrantLock

    # the graph this executable graph was instantiated from, whose nodes can be updated
    # individually (see `update!`)
    const graph::WeakRef

    # memory used by the graph, and by nodes that have been updated individually (see
    # `update!`), leased for as long as the executable graph may be launched
    memory::Vector{Managed}
    node_memory::Dict{CUgraphNode,Vector{Managed}}

    # all that memory, in the order it needs to be locked in when launching the graph
    launch_memory::Vector{Managed}

    # allocations made by the graph that it doesn't free itself (see `unfreed_allocations`)
    allocations::Vector{CuPtr{Cvoid}}

    # the stream the graph was last launched on, and the generation of that stream
    stream::Union{Nothing,CuStream}
    generation::Int
end

Base.unsafe_convert(::Type{CUgraphExec}, exec::CuGraphExec) = exec.handle

function Base.show(io::IO, exec::CuGraphExec)
    print(io, "CuGraphExec(")
    @printf(io, "%p", exec.handle)
    print(io, ")")
end

function lease_memory(graph::CuGraph)
    memory = @lock graph.lock collect(graph.memory)
    foreach(lease!, memory)
    return memory
end

launch_order(memory, node_memory) =
    locking_order(reduce(vcat, values(node_memory); init=memory))

"""
    instantiate(graph::CuGraph, [flags])::CuGraphExec

Create an executable graph from a graph, which can then be launched many times. This is an
expensive operation, so reuse the executable graph, updating it if needed.

See also: [`launch`](@ref), [`update`](@ref).
"""
function instantiate(graph::CuGraph, flags=0)
    @lock graph.lock instantiate_locked(graph, flags)
end
function instantiate_locked(graph::CuGraph, flags)
    allocations = unfreed_allocations(graph)
    handle_ref = Ref{CUgraphExec}()
    context!(graph.ctx) do
        if driver_version() >= v"11.4"
            # graphs that allocate memory without freeing it can only be launched again
            # after freeing that memory, which this flag makes the graph do itself
            flags |= CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
            res = unchecked_cuGraphInstantiateWithFlags(handle_ref, graph, flags)
            if res == ERROR_NOT_SUPPORTED && !isempty(allocations)
                throw(ArgumentError("A graph that allocates memory can only be instantiated once at a time"))
            end
            res == SUCCESS || throw_api_error(res)
        else
            flags == 0 || error("Graph instantiation flags require CUDA 11.4 or higher")
            error_node = Ref{CUgraphNode}()
            buflen = 256
            buf = Vector{UInt8}(undef, buflen)
            GC.@preserve buf begin
                if driver_version() >= v"11"
                    cuGraphInstantiate_v2(handle_ref, graph, error_node, pointer(buf), buflen)
                else
                    cuGraphInstantiate(handle_ref, graph, error_node, pointer(buf), buflen)
                end
            end
        end
    end

    memory = lease_memory(graph)
    exec = CuGraphExec(handle_ref[], graph.ctx, ReentrantLock(), WeakRef(graph), memory,
                       Dict{CUgraphNode,Vector{Managed}}(), locking_order(memory),
                       allocations, nothing, 0)
    resource_finalizer(exec)
    return exec
end

# the memory that a graph allocated during its last launch, which would leak when updating
# or destroying the executable graph
struct GraphAllocations
    ctx::CuContext
    allocations::Vector{CuPtr{Cvoid}}
    stream::CuStream
    generation::Int
end

function launched_allocations(exec::CuGraphExec)
    (exec.stream === nothing || isempty(exec.allocations)) && return nothing
    GraphAllocations(exec.ctx, exec.allocations, exec.stream, exec.generation)
end

function release_now(graph_allocations::GraphAllocations)
    if on_per_thread_stream(graph_allocations.stream)
        # see `release_now(::Managed)`
        destroy_later(synchronize_and(free_allocations, graph_allocations.ctx),
                      graph_allocations)
        return
    end
    free_allocations(graph_allocations)
end

# free that memory after the launch. like other memory, that's not necessarily on the stream
# of the launch, which may have been destroyed or handed to another task since.
function free_allocations(graph_allocations::GraphAllocations)
    (; ctx, allocations, stream, generation) = graph_allocations
    @lock stream_disposal_lock begin
        stream, ctx = release_stream(stream, ctx, generation)
        context!(ctx) do
            for ptr in allocations
                cuMemFreeAsync(ptr, stream)
            end
        end
    end
    return
end

function release_now(exec::CuGraphExec; destroy=cuGraphExecDestroy)
    context!(exec.ctx) do
        allocations = launched_allocations(exec)
        allocations === nothing || release_now(allocations)
        # (destroying an executable graph that is still executing is allowed)
        destroy(exec)
    end
    # (not reached when destroying failed: the resource coordinator then retains the object,
    # so its memory needs to stay leased)
    foreach(unlease!, exec.memory)
    foreach(memory -> foreach(unlease!, memory), values(exec.node_memory))
    return
end

"""
    launch(exec::CuGraphExec, [stream::CuStream])
    exec([stream::CuStream])

Launch an executable graph, by default on the current task's stream.

Just like other operations, launching a graph takes ownership of the memory it uses, so
that it is safe to use that memory from other tasks, and so that it is only released after
the graph has finished executing.
"""
function launch(exec::CuGraphExec, stream::CuStream=stream())
    @lock exec.lock begin
        with_ordered_managed(exec.launch_memory; stream) do
            cuGraphLaunch(exec, stream)
        end
        exec.stream = stream
        exec.generation = CUDACore.generation(stream)
    end
    return
end
(exec::CuGraphExec)(stream::CuStream=stream()) = launch(exec, stream)

"""
    upload(exec::CuGraphExec, [stream::CuStream])

Upload an executable graph to the device, without executing it. This makes the first launch
of the graph faster, which is otherwise slower than subsequent ones.
"""
function upload(exec::CuGraphExec, stream::CuStream=stream())
    @lock exec.lock context!(exec.ctx) do
        cuGraphUpload(exec, stream)
    end
    return
end

@enum_without_prefix visibility=:public CUgraphExecUpdateResult CU_

"""
    update(exec::CuGraphExec, graph::CuGraph; throw_error::Bool=true)::Bool

Update an executable graph to perform the operations of `graph` instead, which needs to
have the same structure as the graph it was instantiated from, e.g., because it was captured
from the same code. This is much cheaper than instantiating a new executable graph.

Returns whether the update succeeded. Unless `throw_error` is false, an error is thrown if
the update failed.
"""
function update(exec::CuGraphExec, graph::CuGraph; throw_error::Bool=true)
    old_memory, old_node_memory, old_allocations = @lock exec.lock begin
        @lock graph.lock begin
            # prepare the new state of the executable graph, which uses the memory of the
            # new graph, before updating it
            memory = lease_memory(graph)
            allocations, launch_memory, node_memory, result = try
                (unfreed_allocations(graph), locking_order(memory),
                 Dict{CUgraphNode,Vector{Managed}}(),
                 context!(() -> exec_update(exec, graph), exec.ctx))
            catch
                foreach(unlease!, memory)
                rethrow()
            end
            if result != GRAPH_EXEC_UPDATE_SUCCESS
                foreach(unlease!, memory)
                throw_error && error("Could not update the executable graph: $result")
                return false
            end

            # commit the new state (which can't fail)
            old = (exec.memory, exec.node_memory, launched_allocations(exec))
            exec.memory = memory
            exec.node_memory = node_memory
            exec.launch_memory = launch_memory
            exec.allocations = allocations
            exec.stream = nothing
            old
        end
    end

    # clean up the old state
    try
        # (not while a graph is being captured, which freeing memory could end up in)
        old_allocations === nothing || discard(old_allocations)
    finally
        foreach(unlease!, old_memory)
        foreach(memory -> foreach(unlease!, memory), values(old_node_memory))
    end
    return true
end

function exec_update(exec::CuGraphExec, graph::CuGraph)
    # (a failed update is also reported as an error, which we handle ourselves)
    res, result = if driver_version() >= v"12.0"
        info = Ref{CUgraphExecUpdateResultInfo}()
        res = unchecked_cuGraphExecUpdate_v2(exec, graph, info)
        res, info[].result
    else
        error_node = Ref{CUgraphNode}()
        result = Ref{CUgraphExecUpdateResult}()
        res = unchecked_cuGraphExecUpdate(exec, graph, error_node, result)
        res, result[]
    end
    res in (SUCCESS, ERROR_GRAPH_EXEC_UPDATE_FAILURE) || throw_api_error(res)
    return result
end

"""
    update!(f, exec::CuGraphExec, node::CuGraphNode)

Update the parameters of a single operation in an executable graph, by capturing the
operation that `f` performs, which needs to be of the same kind as the existing one (e.g.,
launching the same kernel). This is cheaper than updating the entire graph, and makes it
possible to change the arguments of an operation without capturing the graph again:

```julia
graph = CuGraph()
node = only(capture!(graph) do
    @cuda kernel(a)
end)
exec = instantiate(graph)
exec()

update!(exec, node) do
    @cuda kernel(b)
end
exec()
```

Only kernel launches, memory copies and memory sets can be updated this way.
"""
function update!(f::Function, exec::CuGraphExec, node::CuGraphNode)
    exec.graph.value === node.graph ||
        throw(ArgumentError("Can only update nodes of the graph the executable graph was instantiated from"))
    graph = context!(() -> capture(f), exec.ctx)
    new_nodes = nodes(graph)
    length(new_nodes) == 1 ||
        throw(ArgumentError("Expected a single operation to update a node with, got $(length(new_nodes))"))
    new_node = only(new_nodes)
    type = nodetype(node)
    nodetype(new_node) == type ||
        throw(ArgumentError("Cannot update a $(nodetype(node)) node with a $(nodetype(new_node)) node"))

    # the node will use the memory of the captured operation instead. we don't know which
    # memory the node used originally, so keep that around.
    new_memory = lease_memory(graph)
    old_memory = @lock exec.lock begin
        # prepare the new state of the executable graph before updating it
        node_memory, launch_memory, old_memory = try
            node_memory = copy(exec.node_memory)
            old_memory = get(node_memory, node.handle, Managed[])
            node_memory[node.handle] = new_memory
            launch_memory = launch_order(exec.memory, node_memory)
            @lock node.graph.lock set_params(exec, node, new_node, type)
            node_memory, launch_memory, old_memory
        catch
            foreach(unlease!, new_memory)
            rethrow()
        end

        # commit the new state (which can't fail)
        exec.node_memory = node_memory
        exec.launch_memory = launch_memory
        old_memory
    end
    foreach(unlease!, old_memory)

    # the graph we captured isn't needed anymore, so don't wait for the GC to destroy it
    finalize(graph)
    return
end

# set the parameters of a node in an executable graph to those of a node in another graph
function set_params(exec::CuGraphExec, node::CuGraphNode, new_node::CuGraphNode,
                    type::CUgraphNodeType)
    context!(exec.ctx) do
        if type == CU_GRAPH_NODE_TYPE_KERNEL
            params = Ref{CUDA_KERNEL_NODE_PARAMS}()
            cuGraphKernelNodeGetParams_v2(new_node, params)
            cuGraphExecKernelNodeSetParams_v2(exec, node, params)
        elseif type == CU_GRAPH_NODE_TYPE_MEMCPY
            params = Ref{CUDA_MEMCPY3D}()
            cuGraphMemcpyNodeGetParams(new_node, params)
            cuGraphExecMemcpyNodeSetParams(exec, node, params, exec.ctx)
        elseif type == CU_GRAPH_NODE_TYPE_MEMSET
            params = Ref{CUDA_MEMSET_NODE_PARAMS}()
            cuGraphMemsetNodeGetParams(new_node, params)
            cuGraphExecMemsetNodeSetParams(exec, node, params, exec.ctx)
        else
            throw(ArgumentError("Cannot update $type nodes"))
        end
    end
    return
end


## convenience macro

# the executable graph for a `@captured` call site
mutable struct CapturedGraph
    const lock::ReentrantLock
    exec::Union{Nothing,CuGraphExec}
end
CapturedGraph() = CapturedGraph(ReentrantLock(), nothing)

"""
    for ...
        @captured begin
            # code that executes several kernels or CUDA operations
        end
    end

A convenience macro that captures the operations performed by a block of code into a graph,
and executes that graph, reusing the executable graph from the previous execution of the
same block. If the operations have different parameters than during the previous execution
(e.g., because they use different arrays, or scalar arguments changed), the executable graph
is updated, which is much cheaper than instantiating a new one.

As the code is still executed (and captured) every time, this does not reduce the CPU
overhead of executing it as much as launching a previously instantiated graph does. It can
still improve performance of code that executes many short-running kernels, by avoiding
gaps between the kernels on the GPU.

If capturing fails, e.g., because a library needs to be initialized, the code is executed
normally, and capturing it is tried again.

See also: [`capture`](@ref).
"""
macro captured(ex)
    cache = CapturedGraph()
    quote
        captured($cache) do
            $(esc(ex))
        end
    end
end

function captured(f, cache::CapturedGraph)
    @lock cache.lock begin
        executed = false
        graph = capture(f; throw_error=false)
        if graph === nothing
            # capturing may have failed because of initialization that only needs to
            # happen once, so execute the code normally, and try again
            f()
            executed = true
            graph = capture(f)
        end

        exec = cache.exec
        if exec === nothing || !update(exec, graph; throw_error=false)
            cache.exec = exec = instantiate(graph)
        end
        # the graph isn't needed anymore, so don't wait for the GC to destroy it
        finalize(graph)
        executed || launch(exec)
    end
    return
end

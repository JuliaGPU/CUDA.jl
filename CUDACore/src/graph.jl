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

export CuGraph, CuGraphExec, CuGraphNode, capture, instantiate, launch, update, upload,
       @captured
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

# the capture by `capture` that `stream` is part of, if any. must only be called for
# streams that are being captured.
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


## graphs

"""
    CuGraph()

A graph of GPU operations, typically created by recording operations with
[`capture`](@ref). To execute a graph, [`instantiate`](@ref) it.

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
        finalizer(retire!, obj)
        return obj
    end
end

function CuGraph()
    handle_ref = Ref{CUgraph}()
    cuGraphCreate(handle_ref, 0)
    return CuGraph(handle_ref[])
end

Base.unsafe_convert(::Type{CUgraph}, graph::CuGraph) = graph.handle

function dispose(graph::CuGraph)
    try
        context!(graph.ctx) do
            cuGraphDestroy(graph)
        end
    finally
        foreach(unlease!, graph.memory)
    end
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
        cuGraphGetNodes(graph, handles, count)
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


## graph nodes

"""
    CuGraphNode

A node in a [`CuGraph`](@ref), representing an operation. Nodes are owned by their graph.
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

# capture the operations performed by `f` on the current task's stream into a new graph.
# returns the graph, or `nothing` if capturing failed and `throw_error` is false.
function capture_stream(f; mode::CUstreamCaptureMode, throw_error::Bool)
    ctx = context()
    stream = CUDACore.stream()
    capture = CaptureState(stream)
    handle = Ref{CUgraph}(C_NULL)

    # the capture has to be ended on the thread it was started on
    task = current_task()
    sticky = task.sticky
    task.sticky = true
    try
        # disposing of resources can involve operations that aren't allowed while capturing,
        # so wait for any disposal in progress to finish, and prevent new ones from starting
        @lock drain_lock Threads.atomic_add!(active_captures, 1)
        try
            cuStreamBeginCapture_v2(stream, mode)

            # from here on, the capture needs to be ended
            try
                register!(capture)
                # (so that the task's stream isn't changed while capturing, see `priority!`)
                task_local_storage(f, :CUDA_capture_stream, stream)
            catch err
                try
                    end_capture(capture, handle)
                catch
                    # report the original error
                end
                discard_capture(capture, handle[])
                if !throw_error && unsupported_during_capture(err)
                    return nothing
                end
                rethrow()
            end
            res = try
                end_capture(capture, handle)
            catch
                discard_capture(capture, handle[])
                rethrow()
            end
            if res != SUCCESS
                discard_capture(capture, handle[])
                if !throw_error && res == ERROR_STREAM_CAPTURE_INVALIDATED
                    return nothing
                end
                throw_api_error(res)
            end

            graph = CuGraph(handle[], ctx)
            adopt!(graph, capture)
            return graph
        finally
            Threads.atomic_sub!(active_captures, 1)
        end
    finally
        task.sticky = sticky

        # release resources that were retired while capturing
        drain_retired()
    end
end

# end a capture, returning the result
function end_capture(capture::CaptureState, handle::Ref{CUgraph})
    res = unchecked_cuStreamEndCapture(capture.stream, handle)
    unregister!(capture)
    return res
end

function discard_capture(capture::CaptureState, handle::CUgraph)
    # a failed capture may still have produced a graph, which we don't need
    handle == C_NULL || cuGraphDestroy(handle)
    foreach(unlease!, capture.memory)
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

"""
    capture(f; mode=STREAM_CAPTURE_MODE_THREAD_LOCAL, throw_error=true)::CuGraph

Capture the GPU operations that `f` performs on the current task's stream into a graph,
without executing them. Instantiate the graph to execute it, possibly many times:

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

Memory that is used by captured operations is kept alive by the graph.

Not all operations can be captured. Anything that waits for the GPU, like copying memory
back to the CPU, or that creates library handles, results in a [`CaptureError`](@ref) or a
`CuError`. It's typically a good idea to execute `f` once before capturing it, so that
kernels are compiled and libraries are initialized. When `throw_error` is false, failures
due to unsupported operations are not reported, and `nothing` is returned instead.

By default, only the capturing thread is prohibited from performing operations that are
unsafe during capture. Use `mode=STREAM_CAPTURE_MODE_GLOBAL` to also check other threads,
or `mode=STREAM_CAPTURE_MODE_RELAXED` to disable these checks.

See also: [`instantiate`](@ref).
"""
function capture(f::Function; mode::CUstreamCaptureMode=STREAM_CAPTURE_MODE_THREAD_LOCAL,
                 flags::Union{Nothing,CUstreamCaptureMode}=nothing, throw_error::Bool=true)
    if flags !== nothing
        Base.depwarn("The `flags` keyword argument to `capture` has been renamed to `mode`.",
                     :capture)
        mode = flags
    end
    return capture_stream(f; mode, throw_error)
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

    # memory used by the graph, leased for as long as the executable graph may be launched
    memory::Vector{Managed}

    # that memory, in the order it needs to be locked in when launching the graph
    launch_memory::Vector{Managed}
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
    handle_ref = Ref{CUgraphExec}()
    context!(graph.ctx) do
        if driver_version() >= v"11.4"
            cuGraphInstantiateWithFlags(handle_ref, graph, flags)
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
    exec = CuGraphExec(handle_ref[], graph.ctx, ReentrantLock(), memory,
                       locking_order(memory))
    finalizer(retire!, exec)
    return exec
end

function dispose(exec::CuGraphExec)
    try
        context!(exec.ctx) do
            # (destroying an executable graph that is still executing is allowed)
            cuGraphExecDestroy(exec)
        end
    finally
        foreach(unlease!, exec.memory)
    end
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
    old_memory = @lock exec.lock begin
        @lock graph.lock begin
            # prepare the new state of the executable graph, which uses the memory of the
            # new graph, before updating it
            memory = lease_memory(graph)
            launch_memory, result = try
                (locking_order(memory), context!(() -> exec_update(exec, graph), exec.ctx))
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
            old = exec.memory
            exec.memory = memory
            exec.launch_memory = launch_memory
            old
        end
    end

    # clean up the old state
    foreach(unlease!, old_memory)
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

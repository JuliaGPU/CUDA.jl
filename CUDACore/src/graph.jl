# CUDA graphs
#
# a graph records a sequence of GPU operations, so that it can be launched as a whole, with
# much less CPU overhead than launching each operation separately. graphs are created by
# capturing the operations that are executed on a stream, and can be instantiated into
# executable graphs that can be launched many times.

export CuGraph, CuGraphExec, CuGraphNode, capture, instantiate, launch, update, upload,
       @captured
@public nodes


## graphs

"""
    CuGraph()

A graph of GPU operations, typically created by recording operations with
[`capture`](@ref). To execute a graph, [`instantiate`](@ref) it.
"""
mutable struct CuGraph
    const handle::CUgraph
    const ctx::CuContext

    # the driver requires graphs to be used by one thread at a time
    const lock::ReentrantLock

    function CuGraph(handle::CUgraph, ctx::CuContext=context())
        obj = new(handle, ctx, ReentrantLock())
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

function release_now(graph::CuGraph)
    context!(graph.ctx) do
        cuGraphDestroy(graph)
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
            cuStreamBeginCapture_v2(stream, mode)

            # from here on, the capture needs to be ended
            try
                # (so that the task's stream isn't changed while capturing, see `priority!`)
                task_local_storage(f, :CUDA_capture_stream, stream)
            catch err
                unchecked_cuStreamEndCapture(stream, handle)
                discard_capture(handle[])
                if !throw_error && unsupported_during_capture(err)
                    return nothing
                end
                rethrow()
            end
            res = unchecked_cuStreamEndCapture(stream, handle)
            if res != SUCCESS
                discard_capture(handle[])
                if !throw_error && res == ERROR_STREAM_CAPTURE_INVALIDATED
                    return nothing
                end
                throw_api_error(res)
            end
            return CuGraph(handle[], ctx)
        finally
            end_capture()
        end
    finally
        task.sticky = sticky

        # release resources that were retired while capturing
        drain_retired()
    end
end

# a failed capture may still have produced a graph, which we don't need
discard_capture(handle::CUgraph) = handle == C_NULL || cuGraphDestroy(handle)

# whether an exception was caused by an operation that isn't supported during capture
unsupported_during_capture(err) =
    err isa CaptureError ||
    (err isa CuError && err.code in (ERROR_STREAM_CAPTURE_UNSUPPORTED,
                                     ERROR_STREAM_CAPTURE_INVALIDATED,
                                     ERROR_STREAM_CAPTURE_IMPLICIT,
                                     ERROR_CAPTURED_EVENT))

"""
    capture(f; mode=STREAM_CAPTURE_MODE_RELAXED, throw_error=true)::CuGraph

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

Not all operations can be captured. Anything that waits for the GPU, like copying memory
back to the CPU, or that creates library handles, results in a [`CaptureError`](@ref) or a
`CuError`. It's typically a good idea to execute `f` once before capturing it, so that
kernels are compiled and libraries are initialized. When `throw_error` is false, failures
due to unsupported operations are not reported, and `nothing` is returned instead.

CUDA.jl checks the operations it performs itself, like waiting for the GPU, but by default
doesn't ask the driver to prohibit other operations that are potentially unsafe during
capture. The driver performs those checks per thread rather than per task, so they would
also reject operations by unrelated tasks that happen to run on the capturing thread while
the capturing task is waiting. To have the driver perform these checks anyway, e.g., to
debug a library that doesn't support capture, use `mode=STREAM_CAPTURE_MODE_THREAD_LOCAL`
(which checks the capturing thread, and keeps the task on that thread), or
`mode=STREAM_CAPTURE_MODE_GLOBAL` (which also checks other threads).

See also: [`instantiate`](@ref).
"""
function capture(f::Function; mode::CUstreamCaptureMode=STREAM_CAPTURE_MODE_RELAXED,
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
many times.

See also: [`instantiate`](@ref), [`launch`](@ref), [`update`](@ref).
"""
mutable struct CuGraphExec
    const handle::CUgraphExec
    const ctx::CuContext
    const lock::ReentrantLock
end

Base.unsafe_convert(::Type{CUgraphExec}, exec::CuGraphExec) = exec.handle

function Base.show(io::IO, exec::CuGraphExec)
    print(io, "CuGraphExec(")
    @printf(io, "%p", exec.handle)
    print(io, ")")
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

    exec = CuGraphExec(handle_ref[], graph.ctx, ReentrantLock())
    resource_finalizer(exec)
    return exec
end

function release_now(exec::CuGraphExec)
    context!(exec.ctx) do
        # (destroying an executable graph that is still executing is allowed)
        cuGraphExecDestroy(exec)
    end
    return
end

"""
    launch(exec::CuGraphExec, [stream::CuStream])
    exec([stream::CuStream])

Launch an executable graph, by default on the current task's stream.
"""
function launch(exec::CuGraphExec, stream::CuStream=stream())
    @lock exec.lock cuGraphLaunch(exec, stream)
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
    result = @lock exec.lock @lock graph.lock context!(exec.ctx) do
        exec_update(exec, graph)
    end
    if result != GRAPH_EXEC_UPDATE_SUCCESS
        throw_error && error("Could not update the executable graph: $result")
        return false
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

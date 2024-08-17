export CuGraph, capture, instantiate, CuGraphExec, launch, update,
       capture_status, is_capturing,
       @captured


## graph

@enum_without_prefix CUstreamCaptureMode CU_

"""
    CuGraph([flags])

Create an empty graph for use with low-level graph operations. If you want to create a graph
while directly recording operations, use [`capture`](@ref). For a high-level interface that
also automatically executes the graph, use the [`@captured`](@ref) macro.
"""
mutable struct CuGraph
    handle::CUgraph
    ctx::CuContext

    function CuGraph(flags=STREAM_CAPTURE_MODE_GLOBAL)
        handle_ref = Ref{CUgraph}()
        cuGraphCreate(handle_ref, flags)

        ctx = current_context()
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end

    global function capture(f::Function; flags=STREAM_CAPTURE_MODE_GLOBAL,
                            throw_error::Bool=true)
        # graph capture does not support asynchronous memory operations, so disable the GC
        gc_state = GC.enable(false)
        ctx = current_context()
        obj = nothing
        try
            cuStreamBeginCapture_v2(stream(), flags)
            f()
        finally
            handle_ref = Ref{CUgraph}()
            err = unchecked_cuStreamEndCapture(stream(), handle_ref)
            GC.enable(gc_state)
            if err == ERROR_STREAM_CAPTURE_INVALIDATED && !throw_error
                return nothing
            elseif err != CUDA_SUCCESS
                throw_api_error(err)
            end

            obj = new(handle_ref[], ctx)
            finalizer(unsafe_destroy!, obj)
        end
        return obj::CuGraph
    end
end

"""
    capture([flags], [throw_error::Bool=true]) do
        ...
    end

Capture a graph of CUDA operations. The returned graph can then be instantiated and
executed repeatedly for improved performance.

Note that many operations, like initial kernel compilation or memory allocations,
cannot be captured. To work around this, you can set the `throw_error` keyword to false,
which will cause this function to return `nothing` if such a failure happens. You can
then try to evaluate the function in a regular way, and re-record afterwards.

See also: [`instantiate`](@ref).
"""
capture

function unsafe_destroy!(graph::CuGraph)
    context!(graph.ctx; skip_destroyed=true) do
        cuGraphDestroy(graph)
    end
end

Base.unsafe_convert(::Type{CUgraph}, graph::CuGraph) = graph.handle


## instantiated graph

mutable struct CuGraphExec
    handle::CUgraphExec
    graph::CuGraph
    ctx::CuContext

    global function instantiate(graph::CuGraph, flags=0)
        handle_ref = Ref{CUgraphExec}()

        if driver_version() >= v"12.0"
            cuGraphInstantiateWithFlags(handle_ref, graph, flags)
        else
            flags == 0 || error("Flags are not supported on CUDA < 12.0")

            error_node = Ref{CUgraphNode}()
            buflen = 256
            buf = Vector{UInt8}(undef, buflen)

            GC.@preserve buf begin
                if driver_version() >= v"11.0"
                    cuGraphInstantiate_v2(handle_ref, graph, error_node, pointer(buf), buflen)
                else
                    cuGraphInstantiate(handle_ref, graph, error_node, pointer(buf), buflen)
                end
                diag = String(buf)
                # TODO: how to use these?
            end
        end

        ctx = current_context()
        obj = new(handle_ref[], graph, ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

"""
    instantiate(graph::CuGraph)

Creates an executable graph from a graph. This graph can then be launched, or updated
with an other graph.

See also: [`launch`](@ref), [`update`](@ref).
"""
instantiate

function unsafe_destroy!(exec::CuGraphExec)
    context!(exec.ctx; skip_destroyed=true) do
        cuGraphExecDestroy(exec)
    end
end

Base.unsafe_convert(::Type{CUgraphExec}, exec::CuGraphExec) = exec.handle

"""
    launch(exec::CuGraphExec, [stream::CuStream])

Launches an executable graph, by default in the currently-active stream.
"""
launch(exec::CuGraphExec, stream::CuStream=stream()) = cuGraphLaunch(exec, stream)

@enum_without_prefix CUgraphExecUpdateResult CU_

"""
    update(exec::CuGraphExec, graph::CuGraph; [throw_error::Bool=true])

Check whether an executable graph can be updated with a graph and perform the update if
possible. Returns a boolean indicating whether the update was successful. Unless
`throw_error` is set to false, also throws an error if the update failed.
"""
function update(exec::CuGraphExec, graph::CuGraph; throw_error::Bool=true)
    error_node = Ref{CUgraphNode}()
    update_result = Ref{CUgraphExecUpdateResult}()
    cuGraphExecUpdate(exec, graph, error_node, update_result)
    if update_result[] != GRAPH_EXEC_UPDATE_SUCCESS && !throw_error
        return false
    elseif update_result[] != GRAPH_EXEC_UPDATE_SUCCESS
        error("Could not update the executable graph: $(update_result[])")
    end
    return true
end


## global properties

@enum_without_prefix CUstreamCaptureStatus CU_

function capture_status(stream::CuStream=stream())
    status_ref = Ref{CUstreamCaptureStatus}()
    id_ref = Ref{UInt64}()
    cuStreamGetCaptureInfo(stream, status_ref, id_ref)
    return (status=status_ref[],
            id=(status_ref[] == STREAM_CAPTURE_STATUS_ACTIVE ? id_ref[] : nothing))
end

is_capturing(stream::CuStream=stream()) =
    capture_status(stream).status != STREAM_CAPTURE_STATUS_NONE


## convenience macro

"""
    for ...
        @captured begin
            # code that executes several kernels or CUDA operations
        end
    end

A convenience macro for recording a graph of CUDA operations and automatically cache and
update the execution. This can improve performance when executing kernels in a loop, where
the launch overhead might dominate the execution.

!!! warning

    For this to be effective, the kernels and operations executed inside of the
    captured region should not signficantly change across iterations of the loop. It is
    allowed to, e.g., change kernel arguments or inputs to operations, as this will be
    processed by updating the cached executable graph. However, significant changes will
    result in an instantiation of the graph from scratch, which is an expensive operation.

See also: [`capture`](@ref).
"""
macro captured(ex)
    @gensym exec
    @eval __module__ begin
        const $exec = Ref{CuGraphExec}()
    end
    quote
        executed = false

        # capture
        GC.enable(false)    # avoid memory operations during capture
        graph = try
            capture(throw_error=false) do
                $(esc(ex))
            end
        finally
            GC.enable(true)
        end
        if graph === nothing
            # if the capture failed, this may have been due to JIT compilation.
            # execute the body out of capture, and try capturing again.
            $(esc(ex))

            GC.enable(false)
            graph = try
                # don't tolerate capture failures now so that the user will be informed
                capture(throw_error=true) do
                    $(esc(ex))
                end
            finally
                GC.enable(true)
            end
            executed = true
        end

        # update or instantiate
        if !isassigned($(esc(exec))) || !update($(esc(exec))[], graph; throw_error=false)
            $(esc(exec))[] = instantiate(graph)
        end

        # execute
        executed || launch($(esc(exec))[])
    end
end

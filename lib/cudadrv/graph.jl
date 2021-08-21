export CuGraph, capture, instantiate, CuGraphExec, launch, update,
       capture_status, is_capturing,
       @captured


## graph

@enum_without_prefix CUstreamCaptureMode CU_

mutable struct CuGraph
    handle::CUgraph
    ctx::CuContext

    function CuGraph(flags=STREAM_CAPTURE_MODE_GLOBAL)
        handle_ref = Ref{CUgraph}()
        cuGraphCreate(handle_ref, flags)

        ctx = CuCurrentContext()
        obj = new(handle_ref[], ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
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
    global function capture(f::Function; flags=STREAM_CAPTURE_MODE_GLOBAL, throw_error::Bool=true)
        cuStreamBeginCapture_v2(stream(), flags)

        ctx = CuCurrentContext()
        obj = nothing
        try
            f()
        finally
            handle_ref = Ref{CUgraph}()
            err = unsafe_cuStreamEndCapture(stream(), handle_ref)
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

function unsafe_destroy!(graph::CuGraph)
    @finalize_in_ctx graph.ctx cuGraphDestroy(graph)
end

Base.unsafe_convert(::Type{CUgraph}, graph::CuGraph) = graph.handle


## instantiated graph

mutable struct CuGraphExec
    handle::CUgraphExec
    graph::CuGraph
    ctx::CuContext

    """
        instantiate(graph::CuGraph)

    Creates an executable graph from a graph. This graph can then be launched, or updated
    with an other graph.

    See also: [`launch`](@ref), [`update`](@ref).
    """
    global function instantiate(graph::CuGraph)
        handle_ref = Ref{CUgraphExec}()
        error_node = Ref{CUgraphNode}()
        buflen = 256
        buf = Vector{UInt8}(undef, buflen)

        GC.@preserve buf begin
            cuGraphInstantiate_v2(handle_ref, graph, error_node, pointer(buf), buflen)
            diag = String(buf)
            # TODO: how to use these?
        end

        ctx = CuCurrentContext()
        obj = new(handle_ref[], graph, ctx)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function unsafe_destroy!(exec::CuGraphExec)
    @finalize_in_ctx exec.ctx cuGraphDestroy(exec)
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
        @capture begin
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
        graph = capture(throw_error=false) do
            $(esc(ex))
        end
        if graph === nothing
            # if the capture failed, this may have been due to JIT compilation.
            # execute the body out of capture, and try capturing again.
            $(esc(ex))
            graph = capture(throw_error=true) do
                # don't tolerate capture failures now so that the user will be informed
                $(esc(ex))
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

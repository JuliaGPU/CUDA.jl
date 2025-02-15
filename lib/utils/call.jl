# utilities for calling foreign functionality more conveniently

export with_workspace, with_workspaces

## wrapper for foreign functionality that requires a workspace buffer

"""
    with_workspace([cache], bytesize) do workspace
        ...
    end

Create a GPU workspace vector with size `bytesize` (either a number, or a callable
function), and pass it to the do block. Afterwards, the buffer is freed. If you instead want
to cache the workspace, pass any previous instance as the first argument, which will result
in it getting resized instead.

This helper protects against the rare but real issue of the workspace size getter returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).

See also: [`with_workspaces`](@ref), if you need both a GPU and CPU workspace.
"""
with_workspace(f::Base.Callable, size) = with_workspaces(f, nothing, nothing, size, -1)

with_workspace(f::Base.Callable, cache, size) = with_workspaces(f, cache, nothing, size, -1)

"""
    with_workspaces([cache_gpu], [cache_cpu], size_gpu, size_cpu) do workspace_gpu, workspace_cpu
        ...
    end

Create GPU and CPU workspace vectors with size `bytesize` (either a number, or a callable
function), and pass them to the do block.  Afterwards, the buffers are freed. If you instead
want to cache the workspaces, pass any previous instances as the first arguments, which will
result in them getting resized instead.

This helper protects against the rare but real issue of the workspace size getters returning
different results based on the memory pressure, which might change _after_ initial
allocation of the workspace (which can cause a GC collection).

See also: [`with_workspace`](@ref), if you only need a GPU workspace.
"""
with_workspaces(f::Base.Callable, size_gpu, size_cpu) =
    with_workspaces(f, nothing, nothing, size_gpu, size_cpu)
with_workspaces(f::Base.Callable, cache_gpu, size_gpu, size_cpu) =
    with_workspaces(f, cache_gpu, nothing, size_gpu, size_cpu)

function with_workspaces(f::Base.Callable,
                         cache_gpu::Union{Nothing,AbstractVector{UInt8}},
                         cache_cpu::Union{Nothing,AbstractVector{UInt8}},
                         size_gpu::Union{Integer,Function},
                         size_cpu::Union{Integer,Function})

    get_size_gpu() = Int(isa(size_gpu, Integer) ? size_gpu : size_gpu()::Integer)
    get_size_cpu() = Int(isa(size_cpu, Integer) ? size_cpu : size_cpu()::Integer)

    sz_gpu = get_size_gpu()
    sz_cpu = get_size_cpu()

    workspace_gpu = cache_gpu
    while workspace_gpu === nothing || length(workspace_gpu) < sz_gpu
        if workspace_gpu === nothing
            workspace_gpu = CuVector{UInt8}(undef, sz_gpu)
        else
            resize!(workspace_gpu, sz_gpu)
        end
        sz_gpu = get_size_gpu()
    end
    workspace_gpu = workspace_gpu::CuVector{UInt8}

    # size_cpu == -1 means that we don't need a CPU workspace
    if sz_cpu !== -1
        workspace_cpu = cache_cpu
        if workspace_cpu === nothing || length(workspace_cpu) < sz_cpu
            if workspace_cpu === nothing
                workspace_cpu = Vector{UInt8}(undef, sz_cpu)
            else
                resize!(workspace_cpu, sz_cpu)
            end
        end
        workspace_cpu = workspace_cpu::Vector{UInt8}
    end

    # use & free
    try
        if sz_cpu == -1
            f(workspace_gpu)
        else
            f(workspace_gpu, workspace_cpu)
        end
    finally
        if cache_gpu === nothing
            CUDA.unsafe_free!(workspace_gpu)
        end
    end
end

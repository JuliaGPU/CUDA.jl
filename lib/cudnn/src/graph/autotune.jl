"""
    autotune!(graph, bindings; deterministic=false, math_mode=CUDACore.math_mode(),
              max_workspace=nothing)
    autotune!(graph, bindings::Pair...; kwargs...)

Build `graph` like `build!`, but instead of taking the first execution plan that cuDNN's
heuristics propose, time every plan that satisfies the constraints and keep the fastest.
The plans are executed on `bindings`, so the bound outputs are overwritten. This
synchronizes the task's stream, and cannot be done while capturing a CUDA graph.
"""
autotune!(g::Graph, bindings::AbstractDict; kwargs...) =
    autotune!(Returns(nothing), g, bindings; kwargs...)
autotune!(g::Graph, bindings::Pair...; kwargs...) =
    autotune!(g, IdDict{Tensor,Any}(bindings...); kwargs...)

# `prepare()` is called before each plan is tried, e.g., to restore an output that the
# graph also reads.
function autotune!(prepare, g::Graph, bindings::AbstractDict;
                   mode::cudnnBackendOperationGraphMode_t=CUDNN_OPERATIONGRAPH_MODE_AUTO,
                   kwargs...)
    CUDACore.is_capturing() && throw(CUDACore.CaptureError(
        "cannot autotune a cuDNN graph while capturing, as that executes it; " *
        "run the operation with autotuning enabled once before capturing"))
    build_with!(g, mode) do g, opgraph
        plans = Tuple{BackendDescriptor,Int}[]
        best = 0
        try
            ncfgs = foreach_plan(g, opgraph; kwargs...) do plan, ws
                push!(plans, (plan, ws))
                return false
            end
            isempty(plans) && throw(no_engine_error(g, ncfgs))
            # in order of workspace size, so that ties go to the smallest workspace
            sort!(plans; by=last)
            tensors = [t for t in g.tensors if !t.virtual]
            best = fastest_plan(prepare, plans, tensors, bindings)
            return plans[best]
        finally
            for (i, (plan, _)) in enumerate(plans)
                i == best || unsafe_destroy!(plan)
            end
        end
    end
end

# The index of the fastest of `plans`, which are sorted by workspace size.
function fastest_plan(prepare, plans, tensors, bindings)
    uids, pointers, roots = bind(tensors, bindings)

    # share a single workspace, without the plans that need more than can be allocated
    n = length(plans)
    workspace = nothing
    while workspace === nothing
        try
            workspace = CuVector{UInt8}(undef, last(plans[n]))
        catch e
            e isa OutOfGPUMemoryError || rethrow()
            size = last(plans[n])
            while n > 0 && last(plans[n]) >= size
                n -= 1
            end
            n == 0 && rethrow()
        end
    end

    best, best_time = 0, Inf
    err = nothing
    try
        h = handle()
        stream = CUDACore.stream()
        start, stop = CuEvent(), CuEvent()
        vp = variant_pack(; uids, pointers,
                          workspace=sizeof(workspace) == 0 ? C_NULL : pointer(workspace))
        try
            GC.@preserve roots workspace vp for i in 1:n
                plan = first(plans[i])
                prepare()
                time = try
                    time_plan(h, plan, vp, stream, start, stop, 2best_time)
                catch e
                    e isa CUDNNError || rethrow()
                    err = e
                    continue
                end
                if time < best_time
                    best, best_time = i, time
                end
            end
        finally
            unsafe_destroy!(vp)
        end
    finally
        CUDACore.unsafe_free!(workspace)
    end
    best == 0 && throw(err)
    return best
end

# The fastest of a few executions of `plan`, after a warm-up run, giving up on a plan once it
# is slower than `cutoff`, as some candidates are orders of magnitude slower.
function time_plan(h, plan, vp, stream, start, stop, cutoff)
    cudnnBackendExecute(h, plan.ptr, vp.ptr)
    time = Inf
    for _ in 1:3
        record(start, stream)
        cudnnBackendExecute(h, plan.ptr, vp.ptr)
        record(stop, stream)
        synchronize(stop)
        time = min(time, elapsed(start, stop))
        time > cutoff && break
    end
    return time
end

@public autotune!

# GPUArrays.jl interface

#
# Device functionality
#


## execution


@inline function GPUArrays.launch_heuristic(::CUDABackend, f::F, args::Vararg{Any,N};
                                            elements::Int, elements_per_thread::Int) where {F,N}

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, nothing,
                                                                  nothing)

    # this might not be the final context, since we may tune the workgroupsize
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel = @cuda launch=false f(ctx, args...)

    # launching many large blocks) lowers performance, as observed with broadcast, so cap
    # the block size if we don't have a grid-stride kernel (which would keep the grid small)
    if elements_per_thread > 1
        launch_configuration(kernel.fun)
    else
        launch_configuration(kernel.fun; max_threads=256)
    end
end

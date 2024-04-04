# GPUArrays.jl interface

import KernelAbstractions
import KernelAbstractions: Backend

#
# Device functionality
#


## execution

struct CuArrayBackend <: Backend end

@inline function GPUArrays.launch_heuristic(::CuArrayBackend, f::F, args::Vararg{Any,N};
                                            elements::Int, elements_per_thread::Int) where {F,N}
    kernel = @cuda launch=false f(CuKernelContext(), args...)

    # launching many large blocks) lowers performance, as observed with broadcast, so cap
    # the block size if we don't have a grid-stride kernel (which would keep the grid small)
    if elements_per_thread > 1
        launch_configuration(kernel.fun)
    else
        launch_configuration(kernel.fun; max_threads=256)
    end
end

module KernelAbstractionsExt

using CUDACore
using CUDACore: GPUArrays, PrecompileTools

import KernelAbstractions as KA
import KernelInterface as KI

import Adapt

# TODO: Move AbstractGPUSparseArray stuff out
Adapt.adapt_storage(::KA.CPU, a::Union{CuArray,GPUArrays.AbstractGPUSparseArray}) = Adapt.adapt(Array, a)

## kernel launch

# KernelAbstractions launches kernels through the KernelInterface back-end; tell the
# compiler about statically sized workgroups
function KA.compiler_options(obj::KA.Kernel{CUDABackend})
    if KA.workgroupsize(obj) <: KA.StaticSize
        return (; maxthreads = prod(KA.get(KA.workgroupsize(obj))))
    else
        return (;)
    end
end

## other

Adapt.adapt_storage(to::KA.ConstAdaptor, a::CuDeviceArray) = Base.Experimental.Const(a)

## precompilation

# Precompile host launch code, most of which is shared across kernels.
# KernelAbstractions calls `KI.kernel_function` dynamically, so precompiling a launch
# does not reach that call or `KI.launch` on the returned kernel. The kernel itself
# is not compiled: GPUCompiler's cache is keyed on the target, and a workload without
# a GPU cannot target the user's device.

KA.@kernel function precompile_kernel(A, @Const(B))
    i = KA.@index(Global, Linear)
    @inbounds A[i] = 2 * B[i]
end

PrecompileTools.@compile_workload try
    backend = CUDABackend()
    obj = precompile_kernel(backend)
    Arr = CuArray{Float32, 1, CUDACore.default_memory}
    precompile(Tuple{typeof(Core.kwcall), @NamedTuple{ndrange::Int}, typeof(obj), Arr, Arr})

    # Compute argument types for a 1-d launch using KernelAbstractions internals.
    ndrange, _, iterspace = KA.launch_setup(obj, 1024, nothing)
    ctx = KA.mkcontext(obj, ndrange, iterspace, KA.NDLaunch{KA.narrow_index_type(backend)}())
    A = CuDeviceArray{Float32, 1, CUDACore.AS.Global}(
        reinterpret(Core.LLVMPtr{Float32, CUDACore.AS.Global}, C_NULL), (1024,))
    _, tt = KA.prepare_arguments(backend, obj.f, ctx, (A, A))

    F = typeof(obj.f)
    precompile(Tuple{typeof(KI.kernel_function), CUDABackend, F, Type{tt}})
    # `cufunction` calls this from a closure that inference does not follow.
    precompile(Tuple{typeof(Core.kwcall), @NamedTuple{name::Nothing, always_inline::Bool, fastmath::Bool},
                     typeof(CUDACore.cufunction_link), Core.MethodInstance})
    K = KI.Kernel{CUDABackend, CUDACore.CUDAKernels.CompiledKernel{CUDACore.HostKernel{F, tt}, F}}
    precompile(Tuple{typeof(KI.launch), K, Dims{3}, Dims{3}, Tuple{typeof(ctx), Arr, KA.ConstArg{Arr}}})
catch err
    # KernelAbstractions internals may change.
    @debug "Failed to precompile KernelAbstractions launches" exception = (err, catch_backtrace())
end

end

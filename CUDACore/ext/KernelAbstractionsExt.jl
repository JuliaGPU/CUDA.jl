module KernelAbstractionsExt

using CUDACore
using CUDACore: GPUArrays

import KernelAbstractions as KA

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

end

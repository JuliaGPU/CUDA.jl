module CCCL

using CUDACore
using CUDACore: CUstream, CUresult, CUDA_SUCCESS, CuError, initialize_context
using GPUToolbox
using CEnum: @cenum

using GPUCompiler
using LLVM
using Libdl
using Scratch
using Core: LLVMPtr

import CCCL_C_jll

@public functional
@public reduce, reduce!, inclusive_scan!, exclusive_scan!, segmented_reduce!
@public radix_sort!, radix_sortperm!, radix_sortperm, merge_sort!
@public unique_by_key!, unique_by_key, unique_consecutive!, unique_consecutive
@public unsupported, workspace_size

const initialized = Ref{Bool}(false)
"""
    CCCL.functional() -> Bool

Whether the CCCL C library was loaded and initialized successfully, and a CUDA toolkit for it
to compile against was found (see [`CCCL.toolkit_path`](@ref)).
"""
functional() = initialized[]

# core library
include("libcccl.jl")

include("environment.jl")
include("types.jl")
include("operators.jl")
include("builds.jl")
include("scratch.jl")
include("algorithms.jl")
include("unique.jl")

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return
    CUDACore.functional() || return

    if !CCCL_C_jll.is_available()
        @warn "CCCL is not available for your platform. CCCL.functional() will return false."
        return
    end
    global libcccl = CCCL_C_jll.libcccl_c_parallel_v2

    toolkit[] = find_toolkit()
    if toolkit[] === nothing
        @warn """CCCL needs a CUDA toolkit: the CUDA JLLs do not provide every header it needs, and no local toolkit was found (looked in $(join(toolkit_candidates(), ", "))).
                 Set CUDA_PATH to point to one. CCCL.functional() will return false."""
        return
    end

    configure_hostjit!() || return

    initialized[] = true
end

end

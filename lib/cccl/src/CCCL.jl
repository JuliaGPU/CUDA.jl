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
@public radix_sort!, merge_sort!

const initialized = Ref{Bool}(false)
"""
    CCCL.functional() -> Bool

Whether the CCCL C library was loaded and initialized successfully.
"""
functional() = initialized[]

# core library
include("libcccl.jl")

include("environment.jl")
include("types.jl")
include("operators.jl")
include("builds.jl")
include("algorithms.jl")

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0
    precompiling && return
    CUDACore.functional() || return

    if !CCCL_C_jll.is_available()
        @error "CCCL is not available for your platform"
        return
    end
    global libcccl = CCCL_C_jll.libcccl_c_parallel_v2

    configure_hostjit!()

    initialized[] = true
end

end

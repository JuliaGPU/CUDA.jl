module CUDATools

using CUDACore

# Define our own PUBLIC_NAMES for the Julia 1.10 fallback path in the CUDA.jl shim.
# Mirrors CUDACore's @public but keeps a separate registry (and doesn't export @public).
const PUBLIC_NAMES = Symbol[]
_public_sym(s::Symbol) = s
_public_sym(e::Expr) = e.args[1]  # macrocall: @foo → :@foo
macro public(symbols_expr)
    syms = symbols_expr isa Symbol ? [symbols_expr] :
           symbols_expr.head == :tuple ? Symbol[_public_sym(a) for a in symbols_expr.args] :
           [_public_sym(symbols_expr)]
    append!(PUBLIC_NAMES, syms)
    if VERSION >= v"1.11.0-DEV.469"
        esc(Expr(:public, syms...))
    else
        nothing
    end
end

using CUDACore: retry_reclaim, initialize_context,
                cuProfilerStart, cuProfilerStop, cuCtxSynchronize,
                compiler_config, compile, link, COMPILER_KWARGS,
                CuDim3

using GPUToolbox
using GPUCompiler
using GPUCompiler: CompilerJob, methodinstance
using LLVM
using CEnum: @cenum

using CUDA_Compiler_jll: nvdisasm

import Libdl
import Preferences
using Printf

# Alias CUDACore's runtime resolution (avoids duplicating local-vs-JLL logic)
const CUDA_Runtime = CUDACore.CUDA_Runtime
const local_toolkit = CUDACore.local_toolkit

# Submodules
include("../lib/cupti/CUPTI.jl")
export CUPTI

include("../lib/nvml/NVML.jl")
const has_nvml = NVML.has_nvml
export NVML, has_nvml

# Developer tools
include("reflection.jl")
include("profile.jl")
include("utilities.jl")

end

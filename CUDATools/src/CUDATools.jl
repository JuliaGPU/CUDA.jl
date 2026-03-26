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

using CUDACore: compiler_config, compile, link, COMPILER_KWARGS

using GPUCompiler
using GPUCompiler: CompilerJob, methodinstance
using LLVM

using CUDA_Compiler_jll: nvdisasm

import Preferences
using Printf

using CUPTI
export CUPTI

using NVML
using NVML: has_nvml
export NVML, has_nvml

# Developer tools
include("reflection.jl")
include("profile.jl")
include("utilities.jl")

end

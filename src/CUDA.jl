module CUDA


using Reexport

@reexport using CUDACore

# Forward public names from CUDACore so CUDA.xyz works (exported names handled by @reexport)
if VERSION >= v"1.11.0-DEV.469"
    # On 1.11+, `names` returns public names, so we can forward just those
    for n in names(CUDACore)
        Base.isexported(CUDACore, n) && continue
        n === :CUDACore && continue
        isdefined(CUDACore, n) || continue
        @eval using CUDACore: $n
        @eval $(Expr(:public, n))
    end
else
    # On 1.10, there's no `public` keyword. Use PUBLIC_NAMES for names registered
    # by @public, and scan names(; all=true) for public enum values created by
    # @enum_without_prefix (which bypasses @public but creates const bindings).
    public_names = Set{Symbol}(CUDACore.PUBLIC_NAMES)
    for n in names(CUDACore; all=true)
        isdefined(CUDACore, n) || continue
        val = try getfield(CUDACore, n) catch; continue end
        val isa CUDACore.CEnum.Cenum && Symbol(val) !== n && push!(public_names, n)
    end
    for n in public_names
        Base.isexported(CUDACore, n) && continue
        isdefined(CUDACore, n) || continue
        @eval using CUDACore: $n
    end
end

# Load math libraries so their methods (matmul, rand, etc.) are available
using cuBLAS
using cuSPARSE
using cuSOLVER
using cuFFT
using cuRAND

# Forward cuRAND identifiers
using cuRAND: rand, randn, seed!, rand_logn!, rand_logn, rand_poisson!, rand_poisson
@public rand, randn, seed!, rand_logn!, rand_logn, rand_poisson!, rand_poisson

# Backward compatibility: master exported these as submodule names
Base.@deprecate_binding CUBLAS cuBLAS false
Base.@deprecate_binding CUSPARSE cuSPARSE false
Base.@deprecate_binding CUSOLVER cuSOLVER false
Base.@deprecate_binding CUFFT cuFFT false
Base.@deprecate_binding CURAND cuRAND false
export CUBLAS, CUSPARSE, CUSOLVER, CUFFT, CURAND

# Backward compatibility: CUDA.RNG → cuRAND.NativeRNG
Base.@deprecate_binding RNG cuRAND.NativeRNG false

# Backward compatibility: CUDA.default_rng() → cuRAND.native_rng()
Base.@deprecate default_rng() cuRAND.native_rng() false

const has_cusolvermg = cuSOLVER.has_cusolvermg
export has_cusolvermg

end

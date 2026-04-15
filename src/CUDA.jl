module CUDA


using Reexport

@reexport using CUDACore
@reexport using CUDATools

# Forward public names so CUDA.xyz works (exported names handled by @reexport)
if VERSION >= v"1.11.0-DEV.469"
    # On 1.11+, `names` returns public names, so we can forward just those
    for mod in (CUDACore, CUDATools)
        modname = nameof(mod)
        for n in names(mod)
            Base.isexported(mod, n) && continue
            n === modname && continue
            isdefined(mod, n) || continue
            @eval using $modname: $n
            @eval $(Expr(:public, n))
        end
    end
else
    # On 1.10, there's no `public` keyword. Use PUBLIC_NAMES for names registered
    # by @public, and scan names(; all=true) for public enum values created by
    # @enum_without_prefix (which bypasses @public but creates const bindings).
    for mod in (CUDACore, CUDATools)
        modname = nameof(mod)
        public_names = Set{Symbol}(mod.PUBLIC_NAMES)
        for n in names(mod; all=true)
            isdefined(mod, n) || continue
            val = try getfield(mod, n) catch; continue end
            val isa CUDACore.CEnum.Cenum && Symbol(val) !== n && push!(public_names, n)
        end
        for n in public_names
            Base.isexported(mod, n) && continue
            isdefined(mod, n) || continue
            @eval using $modname: $n
        end
    end
end

# Load math libraries so their methods (matmul, rand, etc.) are available
using cuBLAS
using cuSPARSE
using cuSOLVER
using cuFFT
using cuRAND
export cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND

# Backward compatibility: master exported these as submodule names
Base.@deprecate_binding CUBLAS cuBLAS true
Base.@deprecate_binding CUSPARSE cuSPARSE true
Base.@deprecate_binding CUSOLVER cuSOLVER true
Base.@deprecate_binding CUFFT cuFFT true
Base.@deprecate_binding CURAND cuRAND true

# Forward cuRAND identifiers
using cuRAND: rand, randn, seed!, rand_logn!, rand_logn, rand_poisson!, rand_poisson
@public rand, randn, seed!, rand_logn!, rand_logn, rand_poisson!, rand_poisson
@public RNG, default_rng
const RNG = CUDACore.GPUArrays.RNG{CuArray}
default_rng() = cuRAND.gpuarrays_rng()

Base.@deprecate_binding has_cusolvermg cuSOLVER.has_cusolvermg false

end

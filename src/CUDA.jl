module CUDA


using Reexport

@reexport using CUDACore

# Forward public names from CUDACore so CUDA.xyz works (exported names handled by @reexport)
for n in names(CUDACore)
    Base.isexported(CUDACore, n) && continue
    n === :CUDACore && continue
    isdefined(CUDACore, n) || continue
    @eval using CUDACore: $n
    @eval $(Expr(:public, n))
end

# Load math libraries so their methods (matmul, rand, etc.) are available
using cuBLAS
using cuSPARSE
using cuSOLVER
using cuFFT
using cuRAND

# Forward cuRAND's rand/randn/seed! so CUDA.rand etc. work
using cuRAND: rand, randn, seed!
@public rand, randn, seed!

# Forward cuRAND's rand_logn!/rand_poisson! so CUDA.rand_logn! etc. work
using cuRAND: rand_logn!, rand_poisson!
export rand_logn!, rand_poisson!

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

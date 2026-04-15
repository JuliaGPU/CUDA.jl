module CUDA

using Republic

@republic reexport=true inherit=true begin
    using CUDACore
    using CUDATools
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
Base.@deprecate_binding RNG cuRAND.NativeRNG false
Base.@deprecate default_rng() cuRAND.native_rng() false

Base.@deprecate_binding has_cusolvermg cuSOLVER.has_cusolvermg false

end

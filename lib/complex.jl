# CUDA's complex types are defined in terms of vector types (float2, double2),
# but those seem compatible with Julia's complex numbers, so use those.
const cuFloatComplex = Complex{Float32}
const cuDoubleComplex = Complex{Float64}

# aliases
const cuComplex = cuFloatComplex

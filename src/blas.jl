using CUBLAS

CuFloatArray{T<:CUBLAS.CublasFloat,N} = CuArray{T,N}

# TODO: reduce the conversion overhead

Base.LinAlg.A_mul_B!(c::CuFloatArray, a::CuFloatArray, b::CuFloatArray) =
  CuArray(A_mul_B!(CUBLAS.CuArray.(promote(c, a, b))...))

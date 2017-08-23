using CUBLAS

CUBLASArray{T<:CUBLAS.CublasFloat,N} = CuArray{T,N}

Base.LinAlg.A_mul_B!(c::CUBLASArray, a::CUBLASArray, b::CUBLASArray) =
  CuArray(A_mul_B!(CUDAdrv.CuArray.(promote(c, a, b))...))

Base.LinAlg.A_mul_Bt!(c::CUBLASArray, a::CUBLASArray, b::CUBLASArray) =
  CuArray(A_mul_Bt!(CUDAdrv.CuArray.(promote(c, a, b))...))

Base.LinAlg.At_mul_B!(c::CUBLASArray, a::CUBLASArray, b::CUBLASArray) =
  CuArray(At_mul_B!(CUDAdrv.CuArray.(promote(c, a, b))...))

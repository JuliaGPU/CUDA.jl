using CUBLAS

CUBLASArray{T<:CUBLAS.CublasFloat,N} = CuArray{T,N}

Base.LinAlg.A_mul_B!(c::CUBLASArray, a::CUBLASArray, b::CUBLASArray) =
  CuArray(A_mul_B!(CUDAdrv.CuArray.(promote(c, a, b))...))

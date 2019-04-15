@enum libraryPropertyType::Cint begin
      MAJOR_VERSION = 0
      MINOR_VERSION
      PATCH_LEVEL
end

@enum cudaDataType::Cint begin
      R_16F = 2  # `CUDA_R_16F`, real as a half
      C_16F = 6  # `CUDA_C_16F`, complex as a pair of half numbers
      R_32F = 0  # `CUDA_R_32F`, real as a float
      C_32F = 4  # `CUDA_C_32F`, complex as a pair of float numbers
      R_64F = 1  # `CUDA_R_64F`, real as a double
      C_64F = 5  # `CUDA_C_64F`, complex as a pair of double numbers
      R_8I  = 3  # `CUDA_R_8I`,  real as a signed char
      C_8I  = 7  # `CUDA_C_8I`,  complex as a pair of signed char numbers
      R_8U       # `CUDA_R_8U`,  real as a unsigned char
      C_8U       # `CUDA_C_8U`,  complex as a pair of unsigned char numbers
      R_32I      # `CUDA_R_32I`, real as a signed int
      C_32I      # `CUDA_C_32I`, complex as a pair of signed int numbers
      R_32U      # `CUDA_R_32U`, real as a unsigned int
      C_32U      # `CUDA_C_32U`, complex as a pair of unsigned int numbers
end

# Deprecated functionality

using Base: @deprecate_binding

@deprecate_binding BLAS CUBLAS
@deprecate_binding FFT CUFFT

@deprecate cuzeros CuArrays.zeros
@deprecate cuones CuArrays.ones
@deprecate cufill CuArrays.fill

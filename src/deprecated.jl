# Deprecated functionality

import Base: @deprecate_binding

@deprecate_binding BLAS CUBLAS
@deprecate_binding FFT CUFFT

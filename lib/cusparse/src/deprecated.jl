# Deprecated in v6.6: `mv!` multiplies BSR matrices too, with the generic API from cuSPARSE
# 12.6.3 and with bsrmv before. `mv2!` stays exported until the next breaking release.
export mv2!
@deprecate mv2!(transa::SparseChar, alpha::Number, A::CuSparseMatrixBSR, X::CuVector, beta::Number, Y::CuVector, index::SparseChar) mv!(transa, alpha, A, X, beta, Y, index) false

# Deprecated in v6.6: wrappers named after NVIDIA's generic API (cusolverDnX*) are now
# named after the LAPACK routine, like the other wrappers.

@deprecate Xgeev!(jobvl::Char, jobvr::Char, A::StridedCuMatrix) geev!(jobvl, jobvr, A) false
@deprecate Xgesvdp!(jobz::Char, econ::Bool, A::StridedCuMatrix) gesvdp!(jobz, econ, A) false
@deprecate Xgesvdp!(jobz::Char, econ::Int, A::StridedCuMatrix) gesvdp!(jobz, Bool(econ), A) false
@deprecate Xgesvdr!(jobu::Char, jobv::Char, A::StridedCuMatrix, k::Integer; kwargs...) gesvdr!(jobu, jobv, A, k; kwargs...) false
@deprecate Xsyevdx!(jobz::Char, range::Char, uplo::Char, A::StridedCuMatrix; kwargs...) syevdx!(jobz, range, uplo, A; kwargs...) false
@deprecate XsyevBatched!(jobz::Char, uplo::Char, A::StridedCuArray{<:Any,3}) syevBatched!(jobz, uplo, A) false

# the matrix form read an n×(n⋅batch) matrix as a batch, and returned the eigenvalues as
# one vector. A view with padded columns cannot be reshaped into a strided 3-D array, so
# work on a copy of it.
function XsyevBatched!(jobz::Char, uplo::Char, A::StridedCuMatrix)
    Base.depwarn("`XsyevBatched!(jobz, uplo, A::CuMatrix)` is deprecated, use `syevBatched!(jobz, uplo, reshape(A, n, n, :))` instead.", :XsyevBatched!)
    n = size(A, 1)
    B = A isa CuMatrix ? A : copy(A)
    res = syevBatched!(jobz, uplo, reshape(B, n, n, :))
    jobz == 'N' && return vec(res)
    B === A || copyto!(A, B)
    return vec(res[1]), A
end

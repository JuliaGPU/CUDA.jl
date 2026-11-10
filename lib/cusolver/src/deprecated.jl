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

@deprecate Xpotrf!(uplo::Char, A::StridedCuMatrix) potrf!(uplo, A) false
@deprecate Xpotrs!(uplo::Char, A::StridedCuMatrix, B::StridedCuVecOrMat) potrs!(uplo, A, B) false
@deprecate Xgetrf!(A::StridedCuMatrix, ipiv::CuVector{Int64}) getrf!(A, ipiv) false
@deprecate Xgetrf!(A::StridedCuMatrix) getrf!(A, CuVector{Int64}(undef, minimum(size(A)))) false
@deprecate Xgetrs!(trans::Char, A::StridedCuMatrix, ipiv::CuVector{Int64}, B::StridedCuVecOrMat) getrs!(trans, A, ipiv, B) false
@deprecate Xgeqrf!(A::StridedCuMatrix, tau::CuVector) geqrf!(A, tau) false
@deprecate Xgeqrf!(A::StridedCuMatrix) geqrf!(A) false
@deprecate Xgesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix) gesvd!(jobu, jobvt, A) false
@deprecate Xsyevd!(jobz::Char, uplo::Char, A::StridedCuMatrix) syevd!(jobz, uplo, A) false

# Deprecated in v6.6: the routines for Hermitian matrices share the name of the symmetric
# ones, as in LinearAlgebra.LAPACK.
@deprecate heevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{<:Complex}) syevd!(jobz, uplo, A) false

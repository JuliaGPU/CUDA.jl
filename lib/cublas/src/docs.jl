# Documentation of the wrappers in wrappers.jl, most of which are defined per element type.
#
# The wrappers follow LinearAlgebra.BLAS: character arguments select the variant (`trans`,
# `uplo`, `side`, `diag`), `!` functions overwrite their last array argument, and the
# functions without `!` allocate it. Level-1 functions take the number of elements `n`
# first, and read every array with its own stride.

@public version, handle,
        copy!, scal!, dot, dotc, dotu, nrm2, asum, axpy!, axpby!, rot!, rotg!, rotm!, rotmg!,
        swap!, iamax, iamin,
        gemv!, gemv, gemv_batched!, gemv_strided_batched!, gbmv!, gbmv, spmv!, spmv,
        symv!, symv, hemv!, hemv, sbmv!, sbmv, hbmv!, hbmv, tbmv!, tbmv, tbsv!, tbsv,
        trmv!, trmv, trsv!, trsv, ger!, spr!, syr!, her!, her2!,
        gemm!, gemm, gemm_batched!, gemm_batched, gemm_strided_batched!,
        gemm_strided_batched, gemm_grouped_batched!, gemm_grouped_batched,
        symm!, symm, syrk!, syrk, syrkx!, syrkx, hemm!, hemm, herk!, herk, syr2k!, syr2k,
        her2k!, her2k, trmm!, trmm, trsm!, trsm, trsm_batched!, trsm_batched,
        geam!, geam, dgmm!, dgmm,
        getrf_batched!, getrf_batched, getrf_strided_batched!, getrf_strided_batched,
        getrs_batched!, getrs_batched, getrs_strided_batched!, getrs_strided_batched,
        getri_batched!, getri_batched, getri_strided_batched!, matinv_batched,
        geqrf_batched!, geqrf_batched, gels_batched!, gels_batched,
        xt_gemm!, xt_gemm, xt_hemm!, xt_hemm, xt_symm!, xt_symm, xt_syrk!, xt_syrk,
        xt_syrkx!, xt_syrkx, xt_herk!, xt_herk, xt_her2k!, xt_her2k, xt_trmm!, xt_trmm,
        xt_trsm!, xt_trsm, xt_devices, xt_devices!, xt_ndevices


## library

"""
    cuBLAS.version() -> VersionNumber

The version of the cuBLAS library in use.
"""
version

"""
    cuBLAS.handle()

The cuBLAS handle of the current task, for its active context and stream, for calling the
C API of cuBLAS (`cuBLAS.cublas*`) directly. It converts to a `cublasHandle_t`. The handle
is borrowed: don't destroy it, or change its stream, math mode or other settings.
"""
handle


## level 1

"""
    copy!(n, x, y) -> y

Copy the first `n` elements of `x` to `y`. Wraps `cublas<t>copy`; `Float16` and `ComplexF16`
arrays, which cuBLAS can't copy, are copied by a kernel.
"""
copy!

"""
    scal!(n, alpha, x) -> x

Scale the first `n` elements of `x` by `alpha` in place. `alpha` can also be a device-side
scalar (a `CuRef` or one-element `CuArray`). Wraps `cublas<t>scal`.
"""
scal!

"""
    dot(n, x, y) -> T

The dot product of the first `n` elements of the real vectors `x` and `y`. Wraps
`cublas<t>dot`.
"""
dot

"""
    dotc(n, x, y) -> T

The dot product of the first `n` elements of the complex vectors `x` and `y`, conjugating
`x`. Wraps `cublas<t>dotc`.
"""
dotc

"""
    dotu(n, x, y) -> T

The dot product of the first `n` elements of the complex vectors `x` and `y`, without
conjugating either. Wraps `cublas<t>dotu`.
"""
dotu

"""
    nrm2([n,] x) -> real(T)
    nrm2([n,] x, result) -> result

The Euclidean norm of the first `n` elements of `x`, by default all of them. With a
device-side `result` (e.g. a `CuRef`), the norm is written there without synchronizing; this
form does not support `ComplexF16`. Wraps `cublas<t>nrm2`.
"""
nrm2

"""
    asum(n, x) -> real(T)
    asum(n, x, result) -> result

The sum of the absolute values of the first `n` elements of `x` (of the real and imaginary
parts for complex `x`). With a device-side `result`, the sum is written there without
synchronizing. Wraps `cublas<t>asum`.
"""
asum

"""
    axpy!(n, alpha, x, y) -> y

Compute `y = alpha * x + y` for the first `n` elements of `x` and `y`. Wraps
`cublas<t>axpy`.
"""
axpy!

"""
    axpby!(n, alpha, x, beta, y) -> y

Compute `y = alpha * x + beta * y` for the first `n` elements of `x` and `y`, with
`cublas<t>scal` and `cublas<t>axpy`.
"""
axpby!

"""
    rot!(n, x, y, c, s) -> x, y

Apply the plane rotation with cosine `c` and sine `s` to the first `n` elements of `x` and
`y`, in place. Wraps `cublas<t>rot`.
"""
rot!

"""
    rotm!(n, x, y, param) -> x, y

Apply the modified Givens rotation described by the 5-element vector `param` (see
[`rotmg!`](@ref)) to the first `n` elements of `x` and `y`, in place. Only for real arrays. Wraps `cublas<t>rotm`.
"""
rotm!

"""
    swap!(n, x, y) -> x, y

Swap the first `n` elements of `x` and `y`. Wraps `cublas<t>swap`.
"""
swap!

"""
    iamax([n,] x) -> Integer
    iamax([n,] x, result) -> result

The 1-based index of the first element of `x`, among its first `n`, with the largest
absolute value (sum of absolute real and imaginary parts for complex `x`). The index is an
`Int64` from cuBLAS 12, and a `Cint` before. With a device-side `result`, it is written
there without synchronizing. Wraps `cublasI<t>amax`.
"""
iamax

"""
    iamin([n,] x) -> Integer
    iamin([n,] x, result) -> result

Like [`iamax`](@ref), but the index of the element with the smallest absolute value. Wraps
`cublasI<t>amin`.
"""
iamin


## level 2

"""
    gemv!(trans, alpha, A, x, beta, y) -> y

Compute `y = alpha * op(A) * x + beta * y`, where `trans` selects `op` (`'N'`, `'T'` or
`'C'`). Wraps `cublas<t>gemv`.
"""
gemv!

"""
    gemv(trans[, alpha], A, x) -> y

Compute `alpha * op(A) * x`, with `alpha` one by default. See [`gemv!`](@ref).
"""
gemv

"""
    gemv_batched!(trans, alpha, A, x, beta, y) -> y

[`gemv!`](@ref) for each `A[i]`, `x[i]` and `y[i]` of vectors of matrices and vectors of
the same sizes. Requires cuBLAS 11.9.2. Wraps `cublas<t>gemvBatched`.
"""
gemv_batched!

"""
    gemv_strided_batched!(trans, alpha, A, x, beta, y) -> y

[`gemv!`](@ref) for each `A[:, :, i]`, `x[:, i]` and `y[:, i]`. Requires cuBLAS 11.9.2. Wraps
`cublas<t>gemvStridedBatched`.
"""
gemv_strided_batched!

"""
    gbmv!(trans, m, kl, ku, alpha, A, x, beta, y) -> y

Compute `y = alpha * op(A) * x + beta * y` for the `m×n` band matrix `A` with `kl`
subdiagonals and `ku` superdiagonals, stored in LAPACK's band format. Wraps
`cublas<t>gbmv`.
"""
gbmv!

"""
    gbmv(trans, m, kl, ku[, alpha], A, x) -> y

Compute `alpha * op(A) * x` for a band matrix. See [`gbmv!`](@ref).
"""
gbmv

"""
    spmv!(uplo, alpha, AP, x, beta, y) -> y

Compute `y = alpha * A * x + beta * y` for the symmetric matrix `A` whose triangle selected
by `uplo` is stored packed in the vector `AP`. Only for real arrays. Wraps `cublas<t>spmv`.
"""
spmv!

"""
    spmv(uplo[, alpha], AP, x) -> y

Compute `alpha * A * x` for a packed symmetric matrix. See [`spmv!`](@ref).
"""
spmv

"""
    symv!(uplo, alpha, A, x, beta, y) -> y

Compute `y = alpha * A * x + beta * y` for the symmetric matrix `A`, reading the triangle
selected by `uplo`. Wraps `cublas<t>symv`.
"""
symv!

"""
    symv(uplo[, alpha], A, x) -> y

Compute `alpha * A * x` for a symmetric matrix. See [`symv!`](@ref).
"""
symv

"""
    hemv!(uplo, alpha, A, x, beta, y) -> y

Compute `y = alpha * A * x + beta * y` for the Hermitian matrix `A`, reading the triangle
selected by `uplo`. Only for complex arrays. Wraps `cublas<t>hemv`.
"""
hemv!

"""
    hemv(uplo[, alpha], A, x) -> y

Compute `alpha * A * x` for a Hermitian matrix. See [`hemv!`](@ref).
"""
hemv

"""
    sbmv!(uplo, k, alpha, A, x, beta, y) -> y

Compute `y = alpha * A * x + beta * y` for the symmetric band matrix `A` with `k`
super- or subdiagonals, stored in band format. Only for real arrays. Wraps `cublas<t>sbmv`.
"""
sbmv!

"""
    sbmv(uplo, k[, alpha], A, x) -> y

Compute `alpha * A * x` for a symmetric band matrix. See [`sbmv!`](@ref).
"""
sbmv

"""
    hbmv!(uplo, k, alpha, A, x, beta, y) -> y

Compute `y = alpha * A * x + beta * y` for the Hermitian band matrix `A` with `k`
super- or subdiagonals, stored in band format. Only for complex arrays. Wraps `cublas<t>hbmv`.
"""
hbmv!

"""
    hbmv(uplo, k[, alpha], A, x) -> y

Compute `alpha * A * x` for a Hermitian band matrix. See [`hbmv!`](@ref).
"""
hbmv

"""
    tbmv!(uplo, trans, diag, k, A, x) -> x

Compute `x = op(A) * x` for the triangular band matrix `A` with `k` super- or
subdiagonals, stored in band format. `diag = 'U'` means `A` has a unit diagonal. Wraps
`cublas<t>tbmv`.
"""
tbmv!

"""
    tbmv(uplo, trans, diag, k, A, x) -> y

Compute `op(A) * x` for a triangular band matrix. See [`tbmv!`](@ref).
"""
tbmv

"""
    tbsv!(uplo, trans, diag, k, A, x) -> x

Solve `op(A) * y = x` in place in `x`, for the triangular band matrix `A` with `k` super-
or subdiagonals. Wraps `cublas<t>tbsv`.
"""
tbsv!

"""
    tbsv(uplo, trans, diag, k, A, x) -> y

Solve `op(A) * y = x` for a triangular band matrix. See [`tbsv!`](@ref).
"""
tbsv

"""
    trmv!(uplo, trans, diag, A, x) -> x

Compute `x = op(A) * x` for the triangular matrix `A`, reading the triangle selected by
`uplo`. `diag = 'U'` means `A` has a unit diagonal. Wraps `cublas<t>trmv`.
"""
trmv!

"""
    trmv(uplo, trans, diag, A, x) -> y

Compute `op(A) * x` for a triangular matrix. See [`trmv!`](@ref).
"""
trmv

"""
    trsv!(uplo, trans, diag, A, x) -> x

Solve `op(A) * y = x` in place in `x`, for the triangular matrix `A`. Wraps
`cublas<t>trsv`.
"""
trsv!

"""
    trsv(uplo, trans, diag, A, x) -> y

Solve `op(A) * y = x` for a triangular matrix. See [`trsv!`](@ref).
"""
trsv

"""
    ger!(alpha, x, y, A) -> A

Compute the rank-1 update `A = alpha * x * y' + A`. For complex matrices, `y` is
conjugated. Wraps `cublas<t>ger` (`gerc` for complex matrices).
"""
ger!

"""
    spr!(uplo, alpha, x, AP) -> AP

Compute the rank-1 update `A = alpha * x * transpose(x) + A` of the symmetric matrix `A`
whose triangle selected by `uplo` is stored packed in the vector `AP`. Only for real arrays.
Wraps `cublas<t>spr`.
"""
spr!

"""
    syr!(uplo, alpha, x, A) -> A

Compute the rank-1 update `A = alpha * x * transpose(x) + A` of the triangle of the
symmetric matrix `A` selected by `uplo`. Wraps `cublas<t>syr`.
"""
syr!

"""
    her!(uplo, alpha, x, A) -> A

Compute the rank-1 update `A = alpha * x * x' + A` of the triangle of the Hermitian matrix
`A` selected by `uplo`, with real `alpha`. Only for complex arrays. Wraps `cublas<t>her`.
"""
her!

"""
    her2!(uplo, alpha, x, y, A) -> A

Compute the rank-2 update `A = alpha * x * y' + conj(alpha) * y * x' + A` of the triangle
of the Hermitian matrix `A` selected by `uplo`. Only for complex arrays. Wraps `cublas<t>her2`.
"""
her2!


## level 3

"""
    gemm(transA, transB[, alpha], A, B; compute_type=nothing, algo=CUBLAS_GEMM_DEFAULT) -> C

Compute `alpha * op(A) * op(B)`, with `alpha` one by default. See [`gemm!`](@ref).
"""
gemm

"""
    gemm_batched(transA, transB[, alpha], A, B; kwargs...) -> C

Allocating version of [`gemm_batched!`](@ref).
"""
gemm_batched

"""
    gemm_strided_batched(transA, transB[, alpha], A, B; kwargs...) -> C

Allocating version of [`gemm_strided_batched!`](@ref).
"""
gemm_strided_batched

"""
    gemm_grouped_batched(transA, transB[, alpha], A, B; kwargs...) -> C

Allocating version of [`gemm_grouped_batched!`](@ref).
"""
gemm_grouped_batched

"""
    symm!(side, uplo, alpha, A, B, beta, C) -> C

Compute `C = alpha * A * B + beta * C` (`side = 'L'`) or `C = alpha * B * A + beta * C`
(`side = 'R'`) for the symmetric matrix `A`, reading the triangle selected by `uplo`. Wraps
`cublas<t>symm`.
"""
symm!

"""
    symm(side, uplo[, alpha], A, B) -> C

Compute `alpha * A * B` or `alpha * B * A` for a symmetric matrix `A`. See
[`symm!`](@ref).
"""
symm

"""
    syrk!(uplo, trans, alpha, A, beta, C) -> C

Compute the rank-k update `C = alpha * A * transpose(A) + beta * C` (`trans = 'N'`) or
`C = alpha * transpose(A) * A + beta * C` (`trans = 'T'`) of the triangle of the symmetric
matrix `C` selected by `uplo`. Wraps `cublas<t>syrk`.
"""
syrk!

"""
    syrk(uplo, trans[, alpha], A) -> C

Compute the triangle of `alpha * A * transpose(A)` or `alpha * transpose(A) * A` selected by
`uplo`. See [`syrk!`](@ref).
"""
syrk

"""
    syrkx!(uplo, trans, alpha, A, B, beta, C) -> C

Like [`syrk!`](@ref), but computing `alpha * op(A) * transpose(op(B)) + beta * C`, for when
that product is known to be symmetric. Wraps `cublas<t>syrkx`.
"""
syrkx!

"""
    syrkx(uplo, trans, alpha, A, beta, B) -> C
    syrkx(uplo, trans, A, B) -> C

Compute `alpha * op(A) * transpose(op(B))`, with `alpha` one by default. See
[`syrkx!`](@ref). `beta` must be zero: the result is accumulated into uninitialized memory.
"""
syrkx

"""
    hemm!(side, uplo, alpha, A, B, beta, C) -> C

Compute `C = alpha * A * B + beta * C` (`side = 'L'`) or `C = alpha * B * A + beta * C`
(`side = 'R'`) for the Hermitian matrix `A`, reading the triangle selected by `uplo`. Only for complex arrays. Wraps
`cublas<t>hemm`.
"""
hemm!

"""
    hemm(side, uplo[, alpha], A, B) -> C

Compute `alpha * A * B` or `alpha * B * A` for a Hermitian matrix `A`. See
[`hemm!`](@ref).
"""
hemm

"""
    herk!(uplo, trans, alpha, A, beta, C) -> C

Compute the rank-k update `C = alpha * A * A' + beta * C` (`trans = 'N'`) or
`C = alpha * A' * A + beta * C` (`trans = 'C'`) of the triangle of the Hermitian matrix `C`
selected by `uplo`, with real `alpha` and `beta`. Only for complex arrays. Wraps `cublas<t>herk`.
"""
herk!

"""
    herk(uplo, trans[, alpha], A) -> C

Compute the triangle of `alpha * A * A'` or `alpha * A' * A` selected by `uplo`. See
[`herk!`](@ref).
"""
herk

"""
    syr2k!(uplo, trans, alpha, A, B, beta, C) -> C

Compute the rank-2k update
`C = alpha * (A * transpose(B) + B * transpose(A)) + beta * C` (`trans = 'N'`), or with the
transposes on the left (`trans = 'T'`), of the triangle of the symmetric matrix `C`
selected by `uplo`. Wraps `cublas<t>syr2k`.
"""
syr2k!

"""
    syr2k(uplo, trans[, alpha], A, B) -> C

Allocating version of [`syr2k!`](@ref).
"""
syr2k

"""
    her2k!(uplo, trans, alpha, A, B, beta, C) -> C

Compute the rank-2k update `C = alpha * A * B' + conj(alpha) * B * A' + beta * C`
(`trans = 'N'`), or with the adjoints on the left (`trans = 'C'`), of the triangle of the
Hermitian matrix `C` selected by `uplo`, with real `beta`. Only for complex arrays. Wraps `cublas<t>her2k`.
"""
her2k!

"""
    her2k(uplo, trans[, alpha], A, B) -> C

Allocating version of [`her2k!`](@ref).
"""
her2k

"""
    trmm!(side, uplo, transa, diag, alpha, A, B, C) -> C

Compute `C = alpha * op(A) * B` (`side = 'L'`) or `C = alpha * B * op(A)` (`side = 'R'`)
for the triangular matrix `A`. Unlike `LinearAlgebra.BLAS.trmm!`, the result is written to
`C`, which can be `B` itself. Wraps `cublas<t>trmm`.
"""
trmm!

"""
    trmm(side, uplo, transa, diag, alpha, A, B) -> C

Allocating version of [`trmm!`](@ref).
"""
trmm

"""
    trsm!(side, uplo, transa, diag, alpha, A, B) -> B

Solve `op(A) * X = alpha * B` (`side = 'L'`) or `X * op(A) = alpha * B` (`side = 'R'`) for
the triangular matrix `A`, overwriting `B` with `X`. Wraps `cublas<t>trsm`.
"""
trsm!

"""
    trsm(side, uplo, transa, diag, alpha, A, B) -> X

Allocating version of [`trsm!`](@ref).
"""
trsm

"""
    trsm_batched!(side, uplo, transa, diag, alpha, A, B) -> B

[`trsm!`](@ref) for each `A[i]` and `B[i]` of vectors of matrices of the same sizes. Wraps
`cublas<t>trsmBatched`.
"""
trsm_batched!

"""
    trsm_batched(side, uplo, transa, diag, alpha, A, B) -> X

Allocating version of [`trsm_batched!`](@ref).
"""
trsm_batched


## extensions

"""
    geam!(transa, transb, alpha, A, beta, B, C) -> C

Compute `C = alpha * op(A) + beta * op(B)`. Wraps `cublas<t>geam`.
"""
geam!

"""
    geam(transa, transb, alpha, A, beta, B) -> C
    geam(transa, transb, A, B) -> C

Compute `alpha * op(A) + beta * op(B)`, with `alpha` and `beta` one by default. See
[`geam!`](@ref).
"""
geam

"""
    dgmm!(mode, A, X, C) -> C

Compute `C = A * Diagonal(X)` (`mode = 'R'`) or `C = Diagonal(X) * A` (`mode = 'L'`). Wraps
`cublas<t>dgmm`.
"""
dgmm!

"""
    dgmm(mode, A, X) -> C

Allocating version of [`dgmm!`](@ref).
"""
dgmm

"""
    getrf_batched!(A, pivot[, info]) -> pivot, info, A

Compute the LU factorization of each square matrix in the vector `A`, in place. `pivot` is
`true` for partial pivoting, `false` for none, or an `n×batch` `CuMatrix{Cint}` to store the
pivots in. The pivots are returned as such a matrix, with one column per matrix; without
pivoting it holds zeros, which [`getrs_batched!`](@ref) and [`getri_batched!`](@ref) can't
use. `info[i] > 0` means that `A[i]` is singular. Wraps `cublas<t>getrfBatched`.
"""
getrf_batched!

"""
    getrf_batched(A, pivot[, info]) -> pivot, info, A

Like [`getrf_batched!`](@ref), but on a copy of `A`.
"""
getrf_batched

"""
    getrf_strided_batched!(A, pivot[, info]) -> pivot, info, A

[`getrf_batched!`](@ref) for each matrix `A[:, :, i]` of a 3-D array.
"""
getrf_strided_batched!

"""
    getrf_strided_batched(A, pivot[, info]) -> pivot, info, A

Like [`getrf_strided_batched!`](@ref), but on a copy of `A`.
"""
getrf_strided_batched

"""
    getrs_batched!(trans, A, B[, pivot]) -> info, B

Solve `op(A[i]) * X = B[i]` in place in each `B[i]`, using the LU factorizations and pivots
computed by [`getrf_batched!`](@ref) (leave `pivot` out for factorizations without
pivoting). `info` is a host `Ref{Cint}`, nonzero for an invalid argument. Wraps
`cublas<t>getrsBatched`.
"""
getrs_batched!

"""
    getrs_batched(trans, A, B[, pivot]) -> info, X

Like [`getrs_batched!`](@ref), but on a copy of `B`.
"""
getrs_batched

"""
    getrs_strided_batched!(trans, A, B[, pivot]) -> info, B

[`getrs_batched!`](@ref) for each `A[:, :, i]` and `B[:, :, i]` of 3-D arrays.
"""
getrs_strided_batched!

"""
    getrs_strided_batched(trans, A, B[, pivot]) -> info, X

Like [`getrs_strided_batched!`](@ref), but on a copy of `B`.
"""
getrs_strided_batched

"""
    getri_batched!(A, C, pivot) -> info

Write the inverse of each matrix to `C[i]`, from the LU factorizations in `A` and pivots
computed by [`getrf_batched!`](@ref) with pivoting. `info[i] > 0` means that `A[i]` is
singular. Wraps `cublas<t>getriBatched`.
"""
getri_batched!

"""
    getri_batched(A, pivot) -> pivot, info, C

Like [`getri_batched!`](@ref), but allocating the inverses `C`.
"""
getri_batched

"""
    getri_strided_batched!(A, C, pivot) -> info

[`getri_batched!`](@ref) for each `A[:, :, i]` and `C[:, :, i]` of 3-D arrays.
"""
getri_strided_batched!

"""
    matinv_batched(A) -> info, C

Invert each matrix in the vector `A`, which must be smaller than 32×32, without a separate
factorization.
`info[i] > 0` means that `A[i]` is singular. Wraps `cublas<t>matinvBatched`.
"""
matinv_batched

"""
    geqrf_batched!(A) -> tau, A

Compute the QR factorization of each matrix in the vector `A` in place, as LAPACK's `geqrf`
does, returning the reflectors' scalar factors as a vector `tau` of vectors. Wraps
`cublas<t>geqrfBatched`.
"""
geqrf_batched!

"""
    geqrf_batched(A) -> tau, A

Like [`geqrf_batched!`](@ref), but on a copy of `A`.
"""
geqrf_batched

"""
    gels_batched!(trans, A, C) -> A, C, info

Solve the overdetermined least-squares problems `min ‖C[i] - A[i] * X‖` in place, for each
`m×n` matrix `A[i]` with `m ≥ n` and full rank; `X` overwrites the first `n` rows of `C[i]`.
Only `trans = 'N'` is supported by cuBLAS. `info[i] > 0` means that `A[i]` is rank
deficient. Wraps `cublas<t>gelsBatched`.
"""
gels_batched!

"""
    gels_batched(trans, A, C) -> A, C, info

Like [`gels_batched!`](@ref), but on copies of `A` and `C`.
"""
gels_batched


## cuBLASXt

"""
    xt_devices() -> Vector{CuDevice}

The devices that the `xt_*` wrappers (cuBLASXt) of the current task distribute their work
over; all devices by default.
"""
xt_devices

"""
    xt_devices!(devs::Vector{CuDevice})

Set the devices that the `xt_*` wrappers (cuBLASXt) of the current task distribute their
work over.
"""
xt_devices!

"""
    xt_ndevices() -> Int

The number of devices in [`xt_devices`](@ref).
"""
xt_ndevices

"""
    xt_gemm!(transA, transB, alpha, A, B, beta, C) -> C

[`gemm!`](@ref) through cuBLASXt, which splits the product over the devices in
[`xt_devices`](@ref). The arrays can be in host or device memory, and the call returns when
the result is ready. cuBLASXt does not follow stream ordering: device arrays must be ready
when it is called, e.g. by synchronizing after the operations that compute them. Wraps
`cublasXt<t>gemm`.
"""
xt_gemm!

"""
    xt_gemm(transA, transB[, alpha], A, B) -> C

Allocating version of [`xt_gemm!`](@ref).
"""
xt_gemm

"""
    xt_hemm!(side, uplo, alpha, A, B, beta, C) -> C

[`hemm!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>hemm`.
"""
xt_hemm!

"""
    xt_hemm(side, uplo[, alpha], A, B) -> C

Allocating version of [`xt_hemm!`](@ref).
"""
xt_hemm

"""
    xt_symm!(side, uplo, alpha, A, B, beta, C) -> C

[`symm!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>symm`.
"""
xt_symm!

"""
    xt_symm(side, uplo[, alpha], A, B) -> C

Allocating version of [`xt_symm!`](@ref).
"""
xt_symm

"""
    xt_syrk!(uplo, trans, alpha, A, beta, C) -> C

[`syrk!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>syrk`.
"""
xt_syrk!

"""
    xt_syrk(uplo, trans[, alpha], A) -> C

Allocating version of [`xt_syrk!`](@ref).
"""
xt_syrk

"""
    xt_syrkx!(uplo, trans, alpha, A, B, beta, C) -> C

[`syrkx!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>syrkx`.
"""
xt_syrkx!

"""
    xt_syrkx(uplo, trans, alpha, A, B, beta) -> C

Allocating version of [`xt_syrkx!`](@ref). `beta` must be zero: the result is accumulated
into uninitialized memory.
"""
xt_syrkx

"""
    xt_herk!(uplo, trans, alpha, A, beta, C) -> C

[`herk!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>herk`.
"""
xt_herk!

"""
    xt_herk(uplo, trans[, alpha], A) -> C

Allocating version of [`xt_herk!`](@ref).
"""
xt_herk

"""
    xt_her2k!(uplo, trans, alpha, A, B, beta, C) -> C

[`her2k!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>her2k`.
"""
xt_her2k!

"""
    xt_her2k(uplo, trans[, alpha], A, B) -> C

Allocating version of [`xt_her2k!`](@ref).
"""
xt_her2k

"""
    xt_trmm!(side, uplo, transa, diag, alpha, A, B, C) -> C

[`trmm!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>trmm`.
"""
xt_trmm!

"""
    xt_trmm(side, uplo, transa, diag, alpha, A, B) -> C

Allocating version of [`xt_trmm!`](@ref).
"""
xt_trmm

"""
    xt_trsm!(side, uplo, transa, diag, alpha, A, B) -> B

[`trsm!`](@ref) through cuBLASXt (see [`xt_gemm!`](@ref)). Wraps `cublasXt<t>trsm`.
"""
xt_trsm!

"""
    xt_trsm(side, uplo, transa, diag, alpha, A, B) -> X

Allocating version of [`xt_trsm!`](@ref).
"""
xt_trsm

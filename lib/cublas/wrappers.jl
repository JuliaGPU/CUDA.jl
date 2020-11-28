# high-level functionality
#
# modeled from julia/src/base/linalg/blas.jl
# originally authored by Nick Henderson <nwh@stanford.edu> (2014-08-26, MIT licensed)

function cublasCreate()
  handle_ref = Ref{cublasHandle_t}()
  res = @retry_reclaim err->isequal(err, CUBLAS_STATUS_ALLOC_FAILED) ||
                            isequal(err, CUBLAS_STATUS_NOT_INITIALIZED) begin
    unsafe_cublasCreate_v2(handle_ref)
  end
  if res != CUBLAS_STATUS_SUCCESS
    throw_api_error(res)
  end
  handle_ref[]
end

function cublasXtCreate()
  handle_ref = Ref{cublasXtHandle_t}()
  res = @retry_reclaim err->isequal(err, CUBLAS_STATUS_ALLOC_FAILED) ||
                            isequal(err, CUBLAS_STATUS_NOT_INITIALIZED) begin
    unsafe_cublasXtCreate(handle_ref)
  end
  if res != CUBLAS_STATUS_SUCCESS
    throw_api_error(res)
  end
  handle_ref[]
end

function cublasXtGetBlockDim(handle)
  bd = Ref{Int}()
  cublasXtGetBlockDim(handle, bd)
  bd[]
end

function cublasXtGetPinningMemMode(handle)
  mm = Ref{cublasXtPinningMemMode_t}()
  cublasXtGetPinningMemMode(handle, mm)
  mm[]
end

function cublasGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cublasGetProperty(property, value_ref)
  value_ref[]
end

function version(handle=handle())
  version_ref = Ref{Cint}()
  cublasGetVersion_v2(handle, version_ref)
  major, rem = divrem(version_ref[], 1000)
  minor, patch = divrem(rem, 100)
  VersionNumber(major, minor, patch)
end

function juliaStorageType(T::Type{<:Real}, ct::cublasComputeType_t)
    if ct == CUBLAS_COMPUTE_16F || ct == CUBLAS_COMPUTE_16F_PEDANTIC
        return T == BFloat16 ? BFloat16 : Float16
    elseif ct == CUBLAS_COMPUTE_32F || ct == CUBLAS_COMPUTE_32F_PEDANTIC ||
           ct == CUBLAS_COMPUTE_32F_FAST_16F || ct == CUBLAS_COMPUTE_32F_FAST_16BF ||
           ct == CUBLAS_COMPUTE_32F_FAST_TF32
        return Float32
    elseif ct == CUBLAS_COMPUTE_64F || ct == CUBLAS_COMPUTE_64F_PEDANTIC
        return Float64
    elseif ct == CUBLAS_COMPUTE_32I || ct == CUBLAS_COMPUTE_32I_PEDANTIC
        return Int32
    else
        throw(ArgumentError("Julia type equivalent for compute type $ct does not exist!"))
    end
end

function juliaStorageType(T::Type{<:Complex}, ct::cublasComputeType_t)
    if ct == CUBLAS_COMPUTE_16F || ct == CUBLAS_COMPUTE_16F_PEDANTIC
        return T == Complex{BFloat16} == Complex{BFloat16} : Complex{Float16}
    elseif ct == CUBLAS_COMPUTE_32F || ct == CUBLAS_COMPUTE_32F_PEDANTIC ||
           ct == CUBLAS_COMPUTE_32F_FAST_16F || ct == CUBLAS_COMPUTE_32F_FAST_16BF ||
           ct == CUBLAS_COMPUTE_32F_FAST_TF32
        return Complex{Float32}
    elseif ct == CUBLAS_COMPUTE_64F || ct == CUBLAS_COMPUTE_64F_PEDANTIC
        return Complex{Float64}
    elseif ct == CUBLAS_COMPUTE_32I || ct == CUBLAS_COMPUTE_32I_PEDANTIC
        return Complex{Int32}
    else
        throw(ArgumentError("Julia type equivalent for compute type $ct does not exist!"))
    end
end

# Level 1
## copy
for (fname, elty) in ((:cublasDcopy_v2,:Float64),
                      (:cublasScopy_v2,:Float32),
                      (:cublasZcopy_v2,:ComplexF64),
                      (:cublasCcopy_v2,:ComplexF32))
    @eval begin
        function copy!(n::Integer,
                       x::StridedCuArray{$elty},
                       y::StridedCuArray{$elty},)
              $fname(handle(), n, x, stride(x, 1), y, stride(y, 1))
            y
        end
    end
end

## scal
for (fname, elty) in ((:cublasDscal_v2,:Float64),
                      (:cublasSscal_v2,:Float32),
                      (:cublasZscal_v2,:ComplexF64),
                      (:cublasCscal_v2,:ComplexF32))
    @eval begin
        function scal!(n::Integer,
                       alpha::Number,
                       x::StridedCuArray{$elty})
            $fname(handle(), n, alpha, x, stride(x, 1))
            x
        end
    end
end
# specific variants in case x is complex and alpha is real
for (fname, elty, celty) in ((:cublasCsscal_v2, :Float32, :ComplexF32),
                             (:cublasZdscal_v2, :Float64, :ComplexF64))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::StridedCuArray{$celty})
            $fname(handle(), n, alpha, x, stride(x, 1))
            x
        end
    end
end

## dot, dotc, dotu
for (jname, fname, elty) in ((:dot,:cublasDdot_v2,:Float64),
                             (:dot,:cublasSdot_v2,:Float32),
                             (:dotc,:cublasZdotc_v2,:ComplexF64),
                             (:dotc,:cublasCdotc_v2,:ComplexF32),
                             (:dotu,:cublasZdotu_v2,:ComplexF64),
                             (:dotu,:cublasCdotu_v2,:ComplexF32))
    @eval begin
        function $jname(n::Integer,
                        x::DenseCuArray{$elty},
                        y::DenseCuArray{$elty})
            result = Ref{$elty}()
            $fname(handle(), n, x, stride(x, 1), y, stride(y, 1), result)
            return result[]
        end
    end
end

## nrm2
for (fname, elty, ret_type) in ((:cublasDnrm2_v2,:Float64,:Float64),
                                (:cublasSnrm2_v2,:Float32,:Float32),
                                (:cublasDznrm2_v2,:ComplexF64,:Float64),
                                (:cublasScnrm2_v2,:ComplexF32,:Float32))
    @eval begin
        function nrm2(n::Integer,
                      X::StridedCuArray{$elty})
            result = Ref{$ret_type}()
            $fname(handle(), n, X, stride(X, 1), result)
            return result[]
        end
    end
end
nrm2(x::StridedCuArray) = nrm2(length(x), x)

## asum
for (fname, elty, ret_type) in ((:cublasDasum_v2,:Float64,:Float64),
                                (:cublasSasum_v2,:Float32,:Float32),
                                (:cublasDzasum_v2,:ComplexF64,:Float64),
                                (:cublasScasum_v2,:ComplexF32,:Float32))
    @eval begin
        function asum(n::Integer,
                      x::StridedCuArray{$elty})
            result = Ref{$ret_type}()
            $fname(handle(), n, x, stride(x, 1), result)
            return result[]
        end
    end
end

## axpy
for (fname, elty) in ((:cublasDaxpy_v2,:Float64),
                      (:cublasSaxpy_v2,:Float32),
                      (:cublasZaxpy_v2,:ComplexF64),
                      (:cublasCaxpy_v2,:ComplexF32))
    @eval begin
        function axpy!(n::Integer,
                       alpha::Number,
                       dx::StridedCuArray{$elty},
                       dy::StridedCuArray{$elty})
            $fname(handle(), n, alpha, dx, stride(dx, 1), dy, stride(dy, 1))
            dy
        end
    end
end

## rot
for (fname, elty, sty) in ((:cublasSrot_v2,:Float32,:Number),
                           (:cublasDrot_v2,:Float64,:Number),
                           (:cublasCrot_v2,:ComplexF32,:Number),
                           (:cublasCsrot_v2,:ComplexF32,:Real),
                           (:cublasZrot_v2,:ComplexF64,:Number),
                           (:cublasZdrot_v2,:ComplexF64,:Real))
    @eval begin
        function rot!(n::Integer,
                      x::StridedCuArray{$elty},
                      y::StridedCuArray{$elty},
                      c::Real,
                      s::$sty)
            $fname(handle(), n, x, stride(x, 1), y, stride(y, 1), c, s)
            x, y
        end
    end
end

## swap
for (fname, elty) in ((:cublasSswap_v2,:Float32),
                      (:cublasDswap_v2,:Float64),
                      (:cublasCswap_v2,:ComplexF32),
                      (:cublasZswap_v2,:ComplexF64))
    @eval begin
        function swap!(n::Integer,
                       x::DenseCuArray{$elty},
                       y::DenseCuArray{$elty})
            $fname(handle(), n, x, stride(x, 1), y, stride(y, 1))
            x, y
        end
    end
end

function axpby!(n::Integer,
                alpha::Number,
                dx::StridedCuArray{T},
                beta::Number,
                dy::StridedCuArray{T}) where T <: CublasFloat
            scal!(n, beta, dy)
            axpy!(n, alpha, dx, dy)
            dy
end

## iamax
# TODO: fix iamax in julia base
for (fname, elty) in ((:cublasIdamax_v2,:Float64),
                      (:cublasIsamax_v2,:Float32),
                      (:cublasIzamax_v2,:ComplexF64),
                      (:cublasIcamax_v2,:ComplexF32))
    @eval begin
        function iamax(n::Integer,
                       dx::StridedCuArray{$elty})
            result = Ref{Cint}()
            $fname(handle(), n, dx, stride(dx, 1), result)
            return result[]
        end
    end
end
iamax(dx::DenseCuArray) = iamax(length(dx), dx, 1)

## iamin
# iamin is not in standard blas is a CUBLAS extension
for (fname, elty) in ((:cublasIdamin_v2,:Float64),
                      (:cublasIsamin_v2,:Float32),
                      (:cublasIzamin_v2,:ComplexF64),
                      (:cublasIcamin_v2,:ComplexF32))
    @eval begin
        function iamin(n::Integer,
                       dx::DenseCuArray{$elty},)
            result = Ref{Cint}()
            $fname(handle(), n, dx, stride(dx, 1), result)
            return result[]
        end
    end
end
iamin(dx::DenseCuArray) = iamin(length(dx), dx, 1)

# Level 2
## mv
### gemv
for (fname, elty) in ((:cublasDgemv_v2,:Float64),
                      (:cublasSgemv_v2,:Float32),
                      (:cublasZgemv_v2,:ComplexF64),
                      (:cublasCgemv_v2,:ComplexF32))
    @eval begin
        function gemv!(trans::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       X::StridedCuVector{$elty},
                       beta::Number,
                       Y::DenseCuVector{$elty})
            # handle trans
            m,n = size(A)
            # check dimensions
            length(X) == (trans == 'N' ? n : m) && length(Y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            incy = stride(Y,1)
            $fname(handle(), trans, m, n, alpha, A, lda, X, incx, beta, Y, incy)
            Y
        end
        function gemv(trans::Char, alpha::Number, A::StridedCuMatrix{$elty}, X::StridedCuVector{$elty})
            gemv!(trans, alpha, A, X, zero($elty), similar(X, $elty, size(A, (trans == 'N' ? 1 : 2))))
        end
        function gemv(trans::Char, A::StridedCuMatrix{$elty}, X::StridedCuVector{$elty})
            gemv!(trans, one($elty), A, X, zero($elty), similar(X, $elty, size(A, (trans == 'N' ? 1 : 2))))
        end
    end
end

### (GB) general banded matrix-vector multiplication
for (fname, elty) in ((:cublasDgbmv_v2,:Float64),
                      (:cublasSgbmv_v2,:Float32),
                      (:cublasZgbmv_v2,:ComplexF64),
                      (:cublasCgbmv_v2,:ComplexF32))
    @eval begin
        function gbmv!(trans::Char,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::Number,
                       y::CuVector{$elty})
            n = size(A,2)
            # check dimensions
            length(x) == (trans == 'N' ? n : m) && length(y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
        function gbmv(trans::Char,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            # TODO: fix gbmv bug in julia
            n = size(A,2)
            leny = trans == 'N' ? m : n
            gbmv!(trans, m, kl, ku, alpha, A, x, zero($elty), similar(x, $elty, leny))
        end
        function gbmv(trans::Char,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            gbmv(trans, m, kl, ku, one($elty), A, x)
        end
    end
end

### symv
for (fname, elty) in ((:cublasDsymv_v2,:Float64),
                      (:cublasSsymv_v2,:Float32),
                      (:cublasZsymv_v2,:ComplexF64),
                      (:cublasCsymv_v2,:ComplexF32))
    # Note that the complex symv are not BLAS but auiliary functions in LAPACK
    @eval begin
        function symv!(uplo::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::Number,
                       y::CuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
        function symv(uplo::Char, alpha::Number, A::CuMatrix{$elty}, x::CuVector{$elty})
                symv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function symv(uplo::Char, A::CuMatrix{$elty}, x::CuVector{$elty})
            symv(uplo, one($elty), A, x)
        end
    end
end

### hemv
# TODO: fix chemv_ function call bug in julia
for (fname, elty) in ((:cublasZhemv_v2,:ComplexF64),
                      (:cublasChemv_v2,:ComplexF32))
    @eval begin
        function hemv!(uplo::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::Number,
                       y::CuVector{$elty})
            # TODO: fix dimension check bug in julia
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
        function hemv(uplo::Char, alpha::Number, A::CuMatrix{$elty},
                      x::CuVector{$elty})
            hemv!(uplo, alpha, A, x, zero($elty), similar(x))
        end
        function hemv(uplo::Char, A::CuMatrix{$elty},
                      x::CuVector{$elty})
            hemv(uplo, one($elty), A, x)
        end
    end
end

### sbmv, (SB) symmetric banded matrix-vector multiplication
# cublas only has this for D and S
# TODO: check in julia, blas may not have sbmv for C and Z!
for (fname, elty) in ((:cublasDsbmv_v2,:Float64),
                      (:cublasSsbmv_v2,:Float32))
    @eval begin
        function sbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::Number,
                       y::CuVector{$elty})
            m, n = size(A)
            #if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
        function sbmv(uplo::Char, k::Integer, alpha::Number,
                      A::CuMatrix{$elty}, x::CuVector{$elty})
            n = size(A,2)
            sbmv!(uplo, k, alpha, A, x, zero($elty), similar(x, $elty, n))
        end
        function sbmv(uplo::Char, k::Integer, A::CuMatrix{$elty},
                      x::CuVector{$elty})
            sbmv(uplo, k, one($elty), A, x)
        end
    end
end

### hbmv, (HB) Hermitian banded matrix-vector multiplication
for (fname, elty) in ((:cublasZhbmv_v2,:ComplexF64),
                      (:cublasChbmv_v2,:ComplexF32))
    @eval begin
        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::Number,
                       y::CuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            y
        end
        function hbmv(uplo::Char, k::Integer, alpha::Number,
                      A::CuMatrix{$elty}, x::CuVector{$elty})
            n = size(A,2)
            hbmv!(uplo, k, alpha, A, x, zero($elty), similar(x, $elty, n))
        end
        function hbmv(uplo::Char, k::Integer, A::CuMatrix{$elty},
                      x::CuVector{$elty})
            hbmv(uplo, k, one($elty), A, x)
        end
    end
end

### tbmv, (TB) triangular banded matrix-vector multiplication
for (fname, elty) in ((:cublasStbmv_v2,:Float32),
                      (:cublasDtbmv_v2,:Float64),
                      (:cublasZtbmv_v2,:ComplexF64),
                      (:cublasCtbmv_v2,:ComplexF32))
    @eval begin
        function tbmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end
        function tbmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      k::Integer,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            tbmv!(uplo, trans, diag, k, A, copy(x))
        end
    end
end

### tbsv, (TB) triangular banded matrix solve
for (fname, elty) in ((:cublasStbsv_v2,:Float32),
                      (:cublasDtbsv_v2,:Float64),
                      (:cublasZtbsv_v2,:ComplexF64),
                      (:cublasCtbsv_v2,:ComplexF32))
    @eval begin
        function tbsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            x
        end
        function tbsv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      k::Integer,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            tbsv!(uplo, trans, diag, k, A, copy(x))
        end
    end
end

### trmv, Triangular matrix-vector multiplication
for (fname, elty) in ((:cublasDtrmv_v2,:Float64),
                      (:cublasStrmv_v2,:Float32),
                      (:cublasZtrmv_v2,:ComplexF64),
                      (:cublasCtrmv_v2,:ComplexF32))
    @eval begin
        function trmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
        function trmv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            trmv!(uplo, trans, diag, A, copy(x))
        end
    end
end

### trsv, Triangular matrix-vector solve
for (fname, elty) in ((:cublasDtrsv_v2,:Float64),
                      (:cublasStrsv_v2,:Float32),
                      (:cublasZtrsv_v2,:ComplexF64),
                      (:cublasCtrsv_v2,:ComplexF32))
    @eval begin
        function trsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), uplo, trans, diag, n, A, lda, x, incx)
            x
        end
        function trsv(uplo::Char,
                      trans::Char,
                      diag::Char,
                      A::CuMatrix{$elty},
                      x::CuVector{$elty})
            trsv!(uplo, trans, diag, A, copy(x))
        end
    end
end

### ger
for (fname, elty) in ((:cublasDger_v2,:Float64),
                      (:cublasSger_v2,:Float32),
                      (:cublasZgerc_v2,:ComplexF64),
                      (:cublasCgerc_v2,:ComplexF32))
    @eval begin
        function ger!(alpha::Number,
                      x::CuVector{$elty},
                      y::CuVector{$elty},
                      A::CuMatrix{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            $fname(handle(), m, n, alpha, x, incx, y, incy, A, lda)
            A
        end
    end
end

### syr
# TODO: check calls in julia b/c blas may not define syr for Z and C
for (fname, elty) in ((:cublasDsyr_v2,:Float64),
                      (:cublasSsyr_v2,:Float32),
                      (:cublasZsyr_v2,:ComplexF64),
                      (:cublasCsyr_v2,:ComplexF32))
    @eval begin
        function syr!(uplo::Char,
                      alpha::Number,
                      x::CuVector{$elty},
                      A::CuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            $fname(handle(), uplo, n, alpha, x, incx, A, lda)
            A
        end
    end
end

### her
for (fname, elty) in ((:cublasZher_v2,:ComplexF64),
                      (:cublasCher_v2,:ComplexF32))
    @eval begin
        function her!(uplo::Char,
                      alpha::Number,
                      x::CuVector{$elty},
                      A::CuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            $fname(handle(), uplo, n, alpha, x, incx, A, lda)
            A
        end
    end
end

### her2
for (fname, elty) in ((:cublasZher2_v2,:ComplexF64),
                      (:cublasCher2_v2,:ComplexF32))
    @eval begin
        function her2!(uplo::Char,
                      alpha::Number,
                      x::CuVector{$elty},
                      y::CuVector{$elty},
                      A::CuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            length(y) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            $fname(handle(), uplo, n, alpha, x, incx, y, incy, A, lda)
            A
        end
    end
end

# Level 3
## (GE) general matrix-matrix multiplication
for (fname, elty) in
        ((:cublasDgemm_v2,:Float64),
         (:cublasSgemm_v2,:Float32),
         (:cublasHgemm, :Float16),
         (:cublasZgemm_v2,:ComplexF64),
         (:cublasCgemm_v2,:ComplexF32))
    @eval begin
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::DenseCuVecOrMat{$elty},
                       B::DenseCuVecOrMat{$elty},
                       beta::Number,
                       C::DenseCuVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function gemm(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::DenseCuMatrix{$elty},
                      B::DenseCuMatrix{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end
        function gemm(transA::Char,
                      transB::Char,
                      A::DenseCuMatrix{$elty},
                      B::DenseCuMatrix{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

function gemmExComputeType(TA, TB, TC, m, k, n)
    if TA !== TB
        return nothing
    end
    sig = (TA, TC)

    # gemmEx requires sm_50 or higher
    cap = capability(device())
    if cap < v"5"
        return nothing
    end

    math_mode = CUDA.math_mode()
    reduced_precision = CUDA.math_precision()

    if sig === (Float16, Float16)
        # NOTE: Float16=Float16*Float16 can also happen in 32-bit compute
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F
    end

    if m%4 == 0 && n%4 == 0 && k%4 == 0 && sig === (Int8, Int32)
        # Int32=Int8*Int8 requires m,n,k to be multiples of 4
        # https://forums.developer.nvidia.com/t/cublasgemmex-cant-use-cuda-r-8i-compute-type-on-gtx1080/58100/2
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I
    end

    if math_mode == CUDA.FAST_MATH
        if sig === (Float32, Float32) ||
           sig === (Complex{Float32}, Complex{Float32})
            if reduced_precision === :Float16
                return CUBLAS_COMPUTE_32F_FAST_16F
            elseif reduced_precision === :BFloat16
                return CUBLAS_COMPUTE_32F_FAST_16BF
            elseif reduced_precision === :TensorFloat32
                return CUBLAS_COMPUTE_32F_FAST_TF32
            else
                throw(ArgumentError("Unknown reduced precision type $reduced_precision"))
            end
        end
    end

    if sig === (Float16,  Float16) ||
       sig === (Int8,     Float32) ||
       sig === (Float16,  Float32) ||
       sig === (Float32,  Float32) ||
       sig === (Complex{Int8},    Complex{Float32}) ||
       sig === (Complex{Float32}, Complex{Float32})
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F
    end

    if sig === (Float64, Float64) ||
       sig === (Complex{Float64}, Complex{Float64})
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_64F_PEDANTIC : CUBLAS_COMPUTE_64F
    end

    # BFloat16 support was added in CUDA 11
    if version() >= v"11"
        if sig === (BFloat16, BFloat16) ||
           sig === (BFloat16, Float32)
            return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F
        end
    end

    return nothing
end

function gemmEx!(transA::Char, transB::Char,
                 @nospecialize(alpha::Number),
                 @nospecialize(A::DenseCuVecOrMat),
                 @nospecialize(B::DenseCuVecOrMat),
                 @nospecialize(beta::Number),
                 @nospecialize(C::DenseCuVecOrMat);
                 algo::cublasGemmAlgo_t=CUBLAS_GEMM_DEFAULT)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch(""))
    end
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    computeType = gemmExComputeType(eltype(A), eltype(B), eltype(C), m, k, n)
    isnothing(computeType) &&
        throw(ArgumentError("gemmEx does not support $(eltype(C))=$(eltype(A))*$(eltype(B))"))
    computeT = juliaStorageType(eltype(C), computeType)
    if version() < v"11.0"
        # with CUDA 11, the compute type encodes the math mode.
        # before CUDA 11, it was a plain cudaDataType.
        computeType = convert(cudaDataType, computeT)
    end
    cublasGemmEx(handle(), transA, transB, m, n, k, Ref{computeT}(alpha), A, eltype(A), lda, B,
                 eltype(B), ldb, Ref{computeT}(beta), C, eltype(C), ldc, computeType, algo)
    C
end

# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::Vector{<:CuArray{T}}) where {T}
    ptrs = pointer.(batch)
    return CuArray(ptrs)
end

# create a batch of pointers in device memory from a strided device array
@inline function unsafe_strided_batch(strided::DenseCuArray{T}) where {T}
    batchsize = last(size(strided))
    stride = prod(size(strided)[1:end-1])
    ptrs = [pointer(strided, (i-1)*stride + 1) for i in 1:batchsize]
    return CuArray(ptrs)
end

## (GE) general matrix-matrix multiplication batched
for (fname, elty) in
        ((:cublasDgemmBatched,:Float64),
         (:cublasSgemmBatched,:Float32),
         (:cublasZgemmBatched,:ComplexF64),
         (:cublasCgemmBatched,:ComplexF32))
    @eval begin
        function gemm_batched!(transA::Char,
                               transB::Char,
                               alpha::Number,
                               A::Vector{<:CuMatrix{$elty}},
                               B::Vector{<:CuMatrix{$elty}},
                               beta::Number,
                               C::Vector{<:CuMatrix{$elty}})
            if length(A) != length(B) || length(A) != length(C)
                throw(DimensionMismatch(""))
            end
            for (As,Bs,Cs) in zip(A,B,C)
                m = size(As, transA == 'N' ? 1 : 2)
                k = size(As, transA == 'N' ? 2 : 1)
                n = size(Bs, transB == 'N' ? 2 : 1)
                if m != size(Cs,1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end

            m = size(A[1], transA == 'N' ? 1 : 2)
            k = size(A[1], transA == 'N' ? 2 : 1)
            n = size(B[1], transB == 'N' ? 2 : 1)
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            Cptrs = unsafe_batch(C)
            $fname(handle(), transA, transB, m, n, k, alpha, Aptrs, lda, Bptrs,
                   ldb, beta, Cptrs, ldc, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            C
        end
        function gemm_batched(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::Vector{<:CuMatrix{$elty}},
                      B::Vector{<:CuMatrix{$elty}})
            C = CuMatrix{$elty}[similar( B[1], $elty, (size(A[1], transA == 'N' ? 1 : 2),size(B[1], transB == 'N' ? 2 : 1))) for i in 1:length(A)]
            gemm_batched!(transA, transB, alpha, A, B, zero($elty), C )
        end
        function gemm_batched(transA::Char,
                      transB::Char,
                      A::Vector{<:CuMatrix{$elty}},
                      B::Vector{<:CuMatrix{$elty}})
            gemm_batched(transA, transB, one($elty), A, B)
        end
    end
end


## (GE) general matrix-matrix multiplication strided batched
for (fname, elty) in
        ((:cublasDgemmStridedBatched,:Float64),
         (:cublasSgemmStridedBatched,:Float32),
         (:cublasZgemmStridedBatched,:ComplexF64),
         (:cublasCgemmStridedBatched,:ComplexF32))
    @eval begin
        function gemm_strided_batched!(transA::Char,
                               transB::Char,
                               alpha::Number,
                               A::AbstractArray{$elty, 3}, # allow PermutedDimsArray
                               B::AbstractArray{$elty, 3},
                               beta::Number,
                               C::AbstractArray{$elty, 3})
           m = size(A, transA == 'N' ? 1 : 2)
           k = size(A, transA == 'N' ? 2 : 1)
           n = size(B, transB == 'N' ? 2 : 1)

           @assert size(A, 3) == size(C, 3) || size(A, 3) == 1 "batch size mismatch: A != C"
           @assert size(B, 3) == size(C, 3) || size(B, 3) == 1 "batch size mismatch: B != C"

           if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
               throw(DimensionMismatch(""))
           end
           lda = max(1,stride(A,2))
           ldb = max(1,stride(B,2))
           ldc = max(1,stride(C,2))

           strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
           strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
           strideC = stride(C, 3)
           batchCount = size(C, 3)
           $fname(handle(), transA, transB, m, n, k, alpha, A, lda, strideA, B,
                  ldb, strideB, beta, C, ldc, strideC, batchCount)
           C
        end
        function gemm_strided_batched(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::AbstractArray{$elty, 3},
                      B::AbstractArray{$elty, 3})
            C = similar(B, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), max(size(A, 3), size(B, 3))))
            gemm_strided_batched!(transA, transB, alpha, A, B, zero($elty), C )
        end
        function gemm_strided_batched(transA::Char,
                      transB::Char,
                      A::AbstractArray{$elty, 3},
                      B::AbstractArray{$elty, 3})
            gemm_strided_batched(transA, transB, one($elty), A, B)
        end
    end
end

## (SY) symmetric matrix-matrix and matrix-vector multiplication
for (fname, elty) in ((:cublasDsymm_v2,:Float64),
                      (:cublasSsymm_v2,:Float32),
                      (:cublasZsymm_v2,:ComplexF64),
                      (:cublasCsymm_v2,:ComplexF32))
    # TODO: fix julia dimension checks in symm!
    @eval begin
        function symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty},
                       beta::Number,
                       C::CuMatrix{$elty})
            k, nA = size(A)
            if k != nA throw(DimensionMismatch("Matrix A must be square")) end
            m = side == 'L' ? k : size(B,1)
            n = side == 'L' ? size(B,2) : k
            if m != size(C,1) || n != size(C,2) || k != size(B, side == 'L' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), side, uplo, m, n, alpha, A, lda, B, ldb,
                   beta, C, ldc)
            C
        end
        function symm(side::Char,
                      uplo::Char,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            symm!(side, uplo, alpha, A, B, zero($elty), similar(B))
        end
        function symm(side::Char,
                      uplo::Char,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            symm(side, uplo, one($elty), A, B)
        end
    end
end

## syrk
for (fname, elty) in ((:cublasDsyrk_v2,:Float64),
                      (:cublasSsyrk_v2,:Float32),
                      (:cublasZsyrk_v2,:ComplexF64),
                      (:cublasCsyrk_v2,:ComplexF32))
    @eval begin
        function syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::CuVecOrMat{$elty},
                       beta::Number,
                       C::CuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
    end
end
function syrk(uplo::Char,
              trans::Char,
              alpha::Number,
              A::CuVecOrMat)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    syrk!(uplo, trans, alpha, A, zero(T), similar(A, T, (n, n)))
end
syrk(uplo::Char, trans::Char, A::CuVecOrMat) = syrk(uplo, trans,
                                                              one(eltype(A)),
                                                              A)

for (fname, elty) in ((:cublasDsyrkx,:Float64),
                      (:cublasSsyrkx,:Float32),
                      (:cublasZsyrkx,:ComplexF64),
                      (:cublasCsyrkx,:ComplexF32))
    @eval begin
        function syrkx!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::CuVecOrMat{$elty},
                       B::CuVecOrMat{$elty},
                       beta::Number,
                       C::CuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrkx!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function syrkx(uplo::Char,
              trans::Char,
              alpha::Number,
              A::CuVecOrMat,
              beta::Number,
              B::CuVecOrMat)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    syrkx!(uplo, trans, convert(T,alpha), A, B, convert(T,beta), similar(A, T, (n, n)))
end
syrkx(uplo::Char, trans::Char, A::CuVecOrMat, B::CuVecOrMat) = syrkx(uplo, trans,
                                                                 one(eltype(A)), A,
                                                                 zero(eltype(B)), B)

## hemm
for (fname, elty) in ((:cublasZhemm_v2,:ComplexF64),
                      (:cublasChemm_v2,:ComplexF32))
    @eval begin
        function hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty},
                       beta::Number,
                       C::CuMatrix{$elty})
            mA, nA = size(A)
            m, n = size(B)
            mC, nC = size(C)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if ((m != mC) || (n != nC)) throw(DimensionMismatch("B and C must have same dimensions")) end
            if ((side == 'L') && (mA != m)) throw(DimensionMismatch("")) end
            if ((side == 'R') && (mA != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function hemm(uplo::Char,
                      trans::Char,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            m,n = size(B)
            hemm!( uplo, trans, alpha, A, B, zero($elty), similar(B, $elty, (m,n) ) )
        end
        hemm( uplo::Char, trans::Char, A::CuMatrix{$elty}, B::CuMatrix{$elty}) =
            hemm( uplo, trans, one($elty), A, B)
    end
end

## herk
for (fname, elty) in ((:cublasZherk_v2,:ComplexF64),
                      (:cublasCherk_v2,:ComplexF32))
    @eval begin
        function herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::CuVecOrMat{$elty},
                       beta::Real,
                       C::CuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("herk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
        function herk(uplo::Char, trans::Char, alpha::Real, A::CuVecOrMat{$elty})
            n = size(A, trans == 'N' ? 1 : 2)
            herk!(uplo, trans, alpha, A, zero(real($elty)), similar(A, $elty, (n,n)))
        end
        herk(uplo::Char, trans::Char, A::CuVecOrMat{$elty}) =
            herk(uplo, trans, one(real($elty)), A)
   end
end

## syr2k
for (fname, elty) in ((:cublasDsyr2k_v2,:Float64),
                      (:cublasSsyr2k_v2,:Float32),
                      (:cublasZsyr2k_v2,:ComplexF64),
                      (:cublasCsyr2k_v2,:ComplexF32))
    @eval begin
        function syr2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::CuVecOrMat{$elty},
                        B::CuVecOrMat{$elty},
                        beta::Number,
                        C::CuMatrix{$elty})
            # TODO: check size of B in julia (syr2k!)
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1) throw(DimensionMismatch(
                "Inner dimensions of op(A) and op(B.') must match")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function syr2k(uplo::Char,
               trans::Char,
               alpha::Number,
               A::CuVecOrMat,
               B::CuVecOrMat)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    syr2k!(uplo, trans, convert(T,alpha), A, B, zero(T), similar(A, T, (n, n)))
end
syr2k(uplo::Char, trans::Char, A::CuVecOrMat, B::CuVecOrMat) =
    syr2k(uplo, trans, one(eltype(A)), A, B)

## her2k
for (fname, elty) in ((:cublasZher2k_v2,:ComplexF64),
                       (:cublasCher2k_v2,:ComplexF32))
    @eval begin
        function her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::CuVecOrMat{$elty},
                        B::CuVecOrMat{$elty},
                        beta::Real,
                        C::CuMatrix{$elty})
            # TODO: check size of B in julia (her2k!)
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1)
                throw(DimensionMismatch("Inner dimensions of op(A) and op(B.') must match"))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function her2k(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::CuVecOrMat{$elty},
                       B::CuVecOrMat{$elty})
            n = size(A, trans == 'N' ? 1 : 2)
            her2k!(uplo, trans, alpha, A, B, zero(real($elty)), similar(A, $elty, (n,n)))
        end
        her2k(uplo::Char,
              trans::Char,
              A::CuVecOrMat{$elty},
              B::CuVecOrMat{$elty}) = her2k(uplo, trans, one($elty), A, B)
   end
end

## (TR) Triangular matrix and vector multiplication and solution
for (mmname, smname, elty) in
        ((:cublasDtrmm_v2,:cublasDtrsm_v2,:Float64),
         (:cublasStrmm_v2,:cublasStrsm_v2,:Float32),
         (:cublasZtrmm_v2,:cublasZtrsm_v2,:ComplexF64),
         (:cublasCtrmm_v2,:cublasCtrsm_v2,:ComplexF32))
    @eval begin
        # Note: CUBLAS differs from BLAS API for trmm
        #   BLAS: inplace modification of B
        #   CUBLAS: store result in C
        function trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty},
                       C::CuMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trmm!")) end
            mC, nC = size(C)
            if mC != m || nC != n throw(DimensionMismatch("trmm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $mmname(handle(), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
            C
        end
        function trmm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
        end
        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty})
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            $smname(handle(), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end
        function trsm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            trsm!(side, uplo, transa, diag, alpha, A, copy(B))
        end
    end
end

## (TR) triangular triangular matrix solution batched
for (fname, elty) in
        ((:cublasDtrsmBatched,:Float64),
         (:cublasStrsmBatched,:Float32),
         (:cublasZtrsmBatched,:ComplexF64),
         (:cublasCtrsmBatched,:ComplexF32))
    @eval begin
        function trsm_batched!(side::Char,
                               uplo::Char,
                               transa::Char,
                               diag::Char,
                               alpha::Number,
                               A::Vector{<:CuMatrix{$elty}},
                               B::Vector{<:CuMatrix{$elty}})
            if length(A) != length(B)
                throw(DimensionMismatch(""))
            end
            for (As,Bs) in zip(A,B)
                mA, nA = size(As)
                m,n = size(Bs)
                if mA != nA throw(DimensionMismatch("A must be square")) end
                if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm_batched!")) end
            end

            m,n = size(B[1])
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            $fname(handle(), side, uplo, transa, diag, m, n, alpha, Aptrs, lda, Bptrs, ldb, length(A))
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            B
        end
        function trsm_batched(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              alpha::Number,
                              A::Vector{<:CuMatrix{$elty}},
                              B::Vector{<:CuMatrix{$elty}})
            trsm_batched!(side, uplo, transa, diag, alpha, A, copy(B) )
        end
    end
end

# TODO: julia, tr{m,s}m, Char -> Char
# TODO: julia, trmm!, alpha::Number -> alpha::$elty

# BLAS-like extensions
## geam
for (fname, elty) in ((:cublasDgeam,:Float64),
                      (:cublasSgeam,:Float32),
                      (:cublasZgeam,:ComplexF64),
                      (:cublasCgeam,:ComplexF32))
    @eval begin
        function geam!(transa::Char,
                       transb::Char,
                       alpha::Number,
                       A::CuMatrix{$elty},
                       beta::Number,
                       B::CuMatrix{$elty},
                       C::CuMatrix{$elty})
            mA, nA = size(A)
            mB, nB = size(B)
            m, n = size(C)
            if ((transa == 'N') && ((mA != m) && (nA != n ))) throw(DimensionMismatch("")) end
            if ((transa == 'C' || transa == 'T') && ((nA != m) || (mA != n))) throw(DimensionMismatch("")) end
            if ((transb == 'N') && ((mB != m) || (nB != n ))) throw(DimensionMismatch("")) end
            if ((transb == 'C' || transb == 'T') && ((nB != m) || (mB != n))) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
            C
        end
        function geam(transa::Char,
                      transb::Char,
                      alpha::Number,
                      A::CuMatrix{$elty},
                      beta::Number,
                      B::CuMatrix{$elty})
            m,n = size(B)
            if ((transb == 'T' || transb == 'C'))
                geam!( transa, transb, alpha, A, beta, B, similar(B, $elty, (n,m) ) )
            end
            if (transb == 'N')
                geam!( transa, transb, alpha, A, beta, B, similar(B, $elty, (m,n) ) )
            end
        end
        geam( uplo::Char, trans::Char, A::CuMatrix{$elty}, B::CuMatrix{$elty}) =
            geam( uplo, trans, one($elty), A, one($elty), B)
    end
end

## getrfBatched - performs LU factorizations

for (fname, elty) in
        ((:cublasDgetrfBatched,:Float64),
         (:cublasSgetrfBatched,:Float32),
         (:cublasZgetrfBatched,:ComplexF64),
         (:cublasCgetrfBatched,:ComplexF32))
    @eval begin
        function getrf_batched!(n, ptrs::CuVector{CuPtr{$elty}}, lda, pivot::Bool)
            batchSize = length(ptrs)
            info = CuArray{Cint}(undef, batchSize)
            if pivot
                pivotArray = CuArray{Cint}(undef, (n, batchSize))
                $fname(handle(), n, ptrs, lda, pivotArray, info, batchSize)
            else
                $fname(handle(), n, ptrs, lda, CU_NULL, info, batchSize)
                pivotArray = CUDA.zeros(Cint, (n, batchSize))
            end
            unsafe_free!(ptrs)

            return pivotArray, info
        end
    end
end

function getrf_batched!(A::Vector{<:CuMatrix}, pivot::Bool)
    for As in A
        m,n = size(As)
        if m != n
            throw(DimensionMismatch("All matrices must be square!"))
        end
    end
    m,n = size(A[1])
    lda = max(1,stride(A[1],2))

    Aptrs = unsafe_batch(A)
    return getrf_batched!(n, Aptrs, lda, pivot)..., A
end
getrf_batched(A::Vector{<:CuMatrix}, pivot::Bool) = getrf_batched!(copy(A), pivot)

# CUDA has no strided batched getrf, but we can at least avoid constructing costly views
function getrf_strided_batched!(A::DenseCuArray{<:Any, 3}, pivot::Bool)
    m,n = size(A,1), size(A,2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    lda = max(1,stride(A,2))

    Aptrs = unsafe_strided_batch(A)
    return getrf_batched!(n, Aptrs, lda, pivot)..., A
end
getrf_strided_batched(A::DenseCuArray{<:Any, 3}, pivot::Bool) = getrf_strided_batched!(copy(A), pivot)


## getriBatched - performs batched matrix inversion

for (fname, elty) in
        ((:cublasDgetriBatched,:Float64),
         (:cublasSgetriBatched,:Float32),
         (:cublasZgetriBatched,:ComplexF64),
         (:cublasCgetriBatched,:ComplexF32))
    @eval begin
        function getri_batched(A::Vector{<:CuMatrix{$elty}},
                              pivotArray::CuMatrix{Cint})
            for As in A
                m,n = size(As)
                if m != n
                    throw(DimensionMismatch("All A matrices must be square!"))
                end
            end

            C = CuMatrix{$elty}[similar(A[1]) for i in 1:length(A)]
            n = size(A[1])[1]
            lda = max(1,stride(A[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Cptrs = unsafe_batch(C)
            info = CUDA.zeros(Cint,length(A))
            $fname(handle(), n, Aptrs, lda, pivotArray, Cptrs, ldc, info, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)

            pivotArray, info, C
        end
    end
end

## matinvBatched - performs batched matrix inversion

for (fname, elty) in
        ((:cublasDmatinvBatched,:Float64),
         (:cublasSmatinvBatched,:Float32),
         (:cublasZmatinvBatched,:ComplexF64),
         (:cublasCmatinvBatched,:ComplexF32))
    @eval begin
        function matinv_batched(A::Vector{<:CuMatrix{$elty}})
            for As in A
                m,n = size(As)
                if m != n
                    throw(DimensionMismatch("All A matrices must be square!"))
                end
                if n >= 32
                    throw(ArgumentError("matinv requires all matrices be smaller than 32 x 32"))
                end
            end

            C = CuMatrix{$elty}[similar(A[1]) for i in 1:length(A)]
            n = size(A[1])[1]
            lda = max(1,stride(A[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Cptrs = unsafe_batch(C)
            info = CUDA.zeros(Cint,length(A))
            $fname(handle(), n, Aptrs, lda, Cptrs, ldc, info, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)

            info, C
        end
    end
end

## geqrfBatched - performs batched QR factorizations

for (fname, elty) in
        ((:cublasDgeqrfBatched,:Float64),
         (:cublasSgeqrfBatched,:Float32),
         (:cublasZgeqrfBatched,:ComplexF64),
         (:cublasCgeqrfBatched,:ComplexF32))
    @eval begin
        function geqrf_batched!(A::Vector{<:CuMatrix{$elty}})
            m,n = size(A[1])
            lda = max(1,stride(A[1],2))
            Aptrs = unsafe_batch(A)
            hTauArray = [zeros($elty, min(m,n)) for i in 1:length(A)]
            TauArray = CuArray{$elty,1}[]
            for i in 1:length(A)
                push!(TauArray, CuArray(hTauArray[i]))
            end
            Tauptrs = unsafe_batch(TauArray)
            info    = zero(Cint)
            $fname(handle(), m, n, Aptrs, lda, Tauptrs, [info], length(A))
            unsafe_free!(Tauptrs)

            if info != 0
                throw(ArgumentError,string("Invalid value at ",-info))
            end

            TauArray, A
        end
        function geqrf_batched(A::Vector{<:CuMatrix{$elty}})
            geqrf_batched!(copy(A))
        end
    end
end

## gelsBatched - performs batched least squares

for (fname, elty) in
        ((:cublasDgelsBatched,:Float64),
         (:cublasSgelsBatched,:Float32),
         (:cublasZgelsBatched,:ComplexF64),
         (:cublasCgelsBatched,:ComplexF32))
    @eval begin
        function gels_batched!(trans::Char,
                              A::Vector{<:CuMatrix{$elty}},
                              C::Vector{<:CuMatrix{$elty}})
            if length(A) != length(C)
                throw(DimensionMismatch(""))
            end
            m,n = size(A[1])
            mC,nC = size(C[1])
            if mC != m
                throw(DimensionMismatch("Leading dimensions of arrays must match"))
            end
            for (As,Cs) in zip(A,C)
                ms,ns = size(As)
                mCs,nCs = size(Cs)
                if (size(As) != (m, n)) || (size(Cs) != (mC, nC))
                    throw(DimensionMismatch("Dimensions of batched array entries must be invariant"))
                end
            end
            if m < n
                throw(ArgumentError("System must be overdetermined"))
            end

            nrhs = size(C[1])[2]
            lda = max(1,stride(A[1],2))
            ldc = max(1,stride(A[1],2))
            Aptrs = unsafe_batch(A)
            Cptrs = unsafe_batch(C)
            info  = zero(Cint)
            infoarray = CUDA.zeros(Cint, length(A))
            $fname(handle(), trans, m, n, nrhs, Aptrs, lda, Cptrs, ldc, [info], infoarray, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)

            if info != 0
                throw(ArgumentError,string("Invalid value at ",-info))
            end

            A, C, infoarray
        end
        function gels_batched(trans::Char,
                             A::Vector{<:CuMatrix{$elty}},
                             C::Vector{<:CuMatrix{$elty}})
            gels_batched!(trans, deepcopy(A), deepcopy(C))
        end
    end
end

## dgmm
for (fname, elty) in ((:cublasDdgmm,:Float64),
                      (:cublasSdgmm,:Float32),
                      (:cublasZdgmm,:ComplexF64),
                      (:cublasCdgmm,:ComplexF32))
    @eval begin
        function dgmm!(mode::Char,
                       A::CuMatrix{$elty},
                       X::CuVector{$elty},
                       C::CuMatrix{$elty})
            m, n = size(C)
            mA, nA = size(A)
            lx = length(X)
            if ((mA != m) || (nA != n )) throw(DimensionMismatch("")) end
            if ((mode == 'L') && (lx != m)) throw(DimensionMismatch("")) end
            if ((mode == 'R') && (lx != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            ldc = max(1,stride(C,2))
            $fname(handle(), mode, m, n, A, lda, X, incx, C, ldc)
            C
        end
        function dgmm(mode::Char,
                      A::CuMatrix{$elty},
                      X::CuVector{$elty})
            m,n = size(A)
            dgmm!( mode, A, X, similar(A, $elty, (m,n) ) )
        end
    end
end

# cublasXT

# NOTE: cuBLASXt is a blocking API
# > the cuBLASXt API is still a blocking API from the Host point of view:
# > the data results wherever located will be valid on the call return
# > and no device synchronization is required.
#
# HOWEVER: it does not operate with familiar stream semantics, so
# we need to make sure data is available _before_ calling the API.
# this matters most for the tests, but also for allocating methods.

for (fname, elty) in
        ((:cublasXtSgemm,:Float32),
         (:cublasXtDgemm,:Float64),
         (:cublasXtCgemm,:ComplexF32),
         (:cublasXtZgemm,:ComplexF64))
    @eval begin
        function xt_gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                       B::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                       beta::Number,
                       C::Union{CuVecOrMat{$elty}, VecOrMat{$elty}})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function xt_gemm(transA::Char,
                      transB::Char,
                      alpha::Number,
                      A::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                      B::Union{CuVecOrMat{$elty}, VecOrMat{$elty}})
            xt_gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end
        function xt_gemm(transA::Char,
                      transB::Char,
                      A::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                      B::Union{CuVecOrMat{$elty}, VecOrMat{$elty}})
            xt_gemm(transA, transB, one($elty), A, B)
        end
    end
end

for (fname, elty) in ((:cublasXtZhemm,:ComplexF64),
                      (:cublasXtChemm,:ComplexF32))
    @eval begin
        function xt_hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::Union{Matrix{$elty}, CuMatrix{$elty}},
                       B::Union{Matrix{$elty}, CuMatrix{$elty}},
                       beta::Number,
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            mA, nA = size(A)
            m, n = size(B)
            mC, nC = size(C)
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if ((m != mC) || (n != nC)) throw(DimensionMismatch("B and C must have same dimensions")) end
            if ((side == 'L') && (mA != m)) throw(DimensionMismatch("")) end
            if ((side == 'R') && (mA != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function xt_hemm(uplo::Char,
                      trans::Char,
                      alpha::Number,
                      A::Union{Matrix{$elty}, CuMatrix{$elty}},
                      B::Union{Matrix{$elty}, CuMatrix{$elty}})
            m,n = size(B)
            xt_hemm!( uplo, trans, alpha, A, B, zero($elty), similar(B, $elty, (m,n) ) )
        end
        xt_hemm( uplo::Char, trans::Char, A::Union{Matrix{$elty}, CuMatrix{$elty}}, B::Union{Matrix{$elty}, CuMatrix{$elty}}) = xt_hemm( uplo, trans, one($elty), A, B)
    end
end

for (fname, elty) in ((:cublasXtDsymm,:Float64),
                      (:cublasXtSsymm,:Float32),
                      (:cublasXtZsymm,:ComplexF64),
                      (:cublasXtCsymm,:ComplexF32))
    # TODO: fix julia dimension checks in symm!
    @eval begin
        function xt_symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::Union{Matrix{$elty}, CuMatrix{$elty}},
                       B::Union{Matrix{$elty}, CuMatrix{$elty}},
                       beta::Number,
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            k, nA = size(A)
            if k != nA throw(DimensionMismatch("Matrix A must be square")) end
            m = side == 'L' ? k : size(B,1)
            n = side == 'L' ? size(B,2) : k
            if m != size(C,1) || n != size(C,2) || k != size(B, side == 'L' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), side, uplo, m, n, alpha, A, lda, B, ldb,
                   beta, C, ldc)
            C
        end
        function xt_symm(side::Char,
                      uplo::Char,
                      alpha::Number,
                      A::Union{Matrix{$elty}, CuMatrix{$elty}},
                      B::Union{Matrix{$elty}, CuMatrix{$elty}})
            xt_symm!(side, uplo, alpha, A, B, zero($elty), similar(B))
        end
        function xt_symm(side::Char,
                      uplo::Char,
                      A::Union{Matrix{$elty}, CuMatrix{$elty}},
                      B::Union{Matrix{$elty}, CuMatrix{$elty}})
            xt_symm(side, uplo, one($elty), A, B)
        end
    end
end

for (fname, elty) in ((:cublasXtDsyrk,:Float64),
                      (:cublasXtSsyrk,:Float32),
                      (:cublasXtZsyrk,:ComplexF64),
                      (:cublasXtCsyrk,:ComplexF32))
    @eval begin
        function xt_syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                       beta::Number,
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
    end
end
function xt_syrk(uplo::Char,
              trans::Char,
              alpha::Number,
              A::Union{VecOrMat, CuVecOrMat})
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    xt_syrk!(uplo, trans, alpha, A, zero(T), similar(A, T, (n, n)))
end
xt_syrk(uplo::Char, trans::Char, A::Union{VecOrMat, CuVecOrMat}) =
    xt_syrk(uplo, trans, one(eltype(A)), A)

for (fname, elty) in ((:cublasXtDsyrkx,:Float64),
                      (:cublasXtSsyrkx,:Float32),
                      (:cublasXtZsyrkx,:ComplexF64),
                      (:cublasXtCsyrkx,:ComplexF32))
    @eval begin
        function xt_syrkx!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                       B::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                       beta::Number,
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("xt_syrkx!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
    end
end
function xt_syrkx(uplo::Char,
              trans::Char,
              alpha::Number,
              A::Union{VecOrMat, CuVecOrMat},
              B::Union{VecOrMat, CuVecOrMat},
              beta::Number)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    xt_syrkx!(uplo, trans, alpha, A, B, beta, similar(A, T, (n, n)))
end
xt_syrkx(uplo::Char, trans::Char, A::Union{VecOrMat, CuVecOrMat}, B::Union{VecOrMat, CuVecOrMat}) =
    xt_syrkx(uplo, trans, one(eltype(A)), A, B, zero(eltype(B)))

for (fname, elty) in ((:cublasXtZherk,:ComplexF64),
                      (:cublasXtCherk,:ComplexF32))
    @eval begin
        function xt_herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                       beta::Real,
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("herk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            C
        end
        function xt_herk(uplo::Char, trans::Char, alpha::Real, A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}})
            n = size(A, trans == 'N' ? 1 : 2)
            xt_herk!(uplo, trans, alpha, A, real(zero($elty)), similar(A, $elty, (n,n)))
        end
        xt_herk(uplo::Char, trans::Char, A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}}) =
            xt_herk(uplo, trans, real(one($elty)), A)
   end
end

for (fname, elty) in ((:cublasXtZher2k,:ComplexF64),
                      (:cublasXtCher2k,:ComplexF32))
    @eval begin
        function xt_her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                        B::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                        beta::Real,
                        C::Union{Matrix{$elty}, CuMatrix{$elty}})
            # TODO: check size of B in julia (her2k!)
            m, n = size(C)
            if m != n throw(DimensionMismatch("C must be square")) end
            nA = size(A, trans == 'N' ? 1 : 2)
            nB = size(B, trans == 'N' ? 1 : 2)
            if nA != n throw(DimensionMismatch("First dimension of op(A) must match C")) end
            if nB != n throw(DimensionMismatch("First dimension of op(B.') must match C")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            if k != size(B, trans == 'N' ? 2 : 1)
                throw(DimensionMismatch("Inner dimensions of op(A) and op(B.') must match"))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            C
        end
        function xt_her2k(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                       B::Union{VecOrMat{$elty}, CuVecOrMat{$elty}})
            n = size(A, trans == 'N' ? 1 : 2)
            xt_her2k!(uplo, trans, alpha, A, B, zero(real($elty)), similar(A, $elty, (n,n)))
        end
        xt_her2k(uplo::Char, trans::Char, A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                 B::Union{VecOrMat{$elty}, CuVecOrMat{$elty}}) =
            xt_her2k(uplo, trans, one($elty), A, B)
    end
end

for (mmname, smname, elty) in
        ((:cublasXtDtrmm,:cublasXtDtrsm,:Float64),
         (:cublasXtStrmm,:cublasXtStrsm,:Float32),
         (:cublasXtZtrmm,:cublasXtZtrsm,:ComplexF64),
         (:cublasXtCtrmm,:cublasXtCtrsm,:ComplexF32))
    @eval begin
        # Note: CUBLAS differs from BLAS API for trmm
        #   BLAS: inplace modification of B
        #   CUBLAS: store result in C
        function xt_trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::Union{Matrix{$elty}, CuMatrix{$elty}},
                       B::Union{Matrix{$elty}, CuMatrix{$elty}},
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trmm!")) end
            mC, nC = size(C)
            if mC != m || nC != n throw(DimensionMismatch("trmm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $mmname(xt_handle(), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
            C
        end
        function xt_trmm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::Union{CuMatrix{$elty}, Matrix{$elty}},
                      B::Union{CuMatrix{$elty}, Matrix{$elty}})
            xt_trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
        end
        function xt_trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::Union{CuMatrix{$elty}, Matrix{$elty}},
                       B::Union{CuMatrix{$elty}, Matrix{$elty}})
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            $smname(xt_handle(), side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb)
            B
        end
        function xt_trsm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::Number,
                      A::Union{CuMatrix{$elty}, Matrix{$elty}},
                      B::Union{CuMatrix{$elty}, Matrix{$elty}})
            # TODO: better way to perform synchronous copy
            xt_trsm!(side, uplo, transa, diag, alpha, A, @sync(copy(B)))
        end
    end
end

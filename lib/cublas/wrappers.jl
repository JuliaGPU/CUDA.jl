# high-level functionality
#
# modeled from julia/src/base/linalg/blas.jl
# originally authored by Nick Henderson <nwh@stanford.edu> (2014-08-26, MIT licensed)

function cublasCreate_v2()
  handle = Ref{cublasHandle_t}()
  cublasCreate_v2(handle)
  handle[]
end

function cublasXtCreate()
  handle = Ref{cublasXtHandle_t}()
  cublasXtCreate(handle)
  handle[]
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

function cublasGetVersion_v2(handle)
  version = Ref{Cint}()
  cublasGetVersion_v2(handle, version)
  version[]
end

function cublasGetProperty(property::libraryPropertyType)
  value_ref = Ref{Cint}()
  cublasGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cublasGetProperty(CUDA.MAJOR_VERSION),
                          cublasGetProperty(CUDA.MINOR_VERSION),
                          cublasGetProperty(CUDA.PATCH_LEVEL))

# Level 1
## copy
for (fname, elty) in ((:cublasDcopy_v2,:Float64),
                      (:cublasScopy_v2,:Float32),
                      (:cublasZcopy_v2,:ComplexF64),
                      (:cublasCcopy_v2,:ComplexF32))
    @eval begin
        # SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
        function blascopy!(n::Integer,
                           DX::CuArray{$elty},
                           incx::Integer,
                           DY::CuArray{$elty},
                           incy::Integer)
              $fname(handle(), n, DX, incx, DY, incy)
            DY
        end
    end
end

## scal
for (fname, elty) in ((:cublasDscal_v2,:Float64),
                      (:cublasSscal_v2,:Float32),
                      (:cublasZscal_v2,:ComplexF64),
                      (:cublasCscal_v2,:ComplexF32))
    @eval begin
        # SUBROUTINE DSCAL(N,DA,DX,INCX)
        function scal!(n::Integer,
                       DA::$elty,
                       DX::CuArray{$elty},
                       incx::Integer)
            $fname(handle(), n, Ref(DA), DX, incx)
            DX
        end
    end
end
# In case DX is complex, and DA is real, use dscal/sscal to save flops
for (fname, elty, celty) in ((:cublasSscal_v2, :Float32, :ComplexF32),
                             (:cublasDscal_v2, :Float64, :ComplexF64))
    @eval begin
        # SUBROUTINE DSCAL(N,DA,DX,INCX)
        function scal!(n::Integer,
                       DA::$elty,
                       DX::CuArray{$celty},
                       incx::Integer)
            #DY = reinterpret($elty,DX,(2*n,))
            #$(cublascall(fname))(handle(),2*n,[DA],DY,incx)
            $fname(handle(), 2*n, Ref(DA), DX, incx)
            DX
        end
    end
end

## dot, dotc, dotu
# cublasStatus_t cublasDdot_v2
#   (cublasHandle_t handle,
#    int n,
#    const double *x, int incx,
#    const double *y, int incy,
#    double *result);
for (jname, fname, elty) in ((:dot,:cublasDdot_v2,:Float64),
                             (:dot,:cublasSdot_v2,:Float32),
                             (:dotc,:cublasZdotc_v2,:ComplexF64),
                             (:dotc,:cublasCdotc_v2,:ComplexF32),
                             (:dotu,:cublasZdotu_v2,:ComplexF64),
                             (:dotu,:cublasCdotu_v2,:ComplexF32))
    @eval begin
        function $jname(n::Integer,
                        DX::CuArray{$elty},
                        incx::Integer,
                        DY::CuArray{$elty},
                        incy::Integer)
            result = Ref{$elty}()
            $fname(handle(), n, DX, incx, DY, incy, result)
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
        # SUBROUTINE DNRM2(N,X,INCX)
        function nrm2(n::Integer,
                      X::CuArray{$elty},
                      incx::Integer)
            result = Ref{$ret_type}()
            $fname(handle(), n, X, incx, result)
            return result[]
        end
    end
end
# TODO: consider CuVector and CudaStridedVector
#nrm2(x::StridedVector) = nrm2(length(x), x, stride(x,1))
nrm2(x::CuArray) = nrm2(length(x), x, 1)

## asum
for (fname, elty, ret_type) in ((:cublasDasum_v2,:Float64,:Float64),
                                (:cublasSasum_v2,:Float32,:Float32),
                                (:cublasDzasum_v2,:ComplexF64,:Float64),
                                (:cublasScasum_v2,:ComplexF32,:Float32))
    @eval begin
        # SUBROUTINE ASUM(N, X, INCX)
        function asum(n::Integer,
                      X::CuArray{$elty},
                      incx::Integer)
            result = Ref{$ret_type}()
            $fname(handle(), n, X, incx, result)
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
        # SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
        # DY <- DA*DX + DY
        # cublasStatus_t cublasSaxpy_v2(
        #   cublasHandle_t handle,
        #   int n,
        #   const float *alpha, /* host or device pointer */
        #   const float *x,
        #   int incx,
        #   float *y,
        #   int incy);
        function axpy!(n::Integer,
                       alpha::($elty),
                       dx::CuArray{$elty},
                       incx::Integer,
                       dy::CuArray{$elty},
                       incy::Integer)
            $fname(handle(), n, Ref(alpha), dx, incx, dy, incy)
            dy
        end
    end
end

function axpy!(alpha::Ta,
               x::CuArray{T},
               rx::Union{UnitRange{Ti},AbstractRange{Ti}},
               y::CuArray{T},
               ry::Union{UnitRange{Ti},AbstractRange{Ti}}) where {T<:CublasFloat,Ta<:Number,Ti<:Integer}
    length(rx)==length(ry) || throw(DimensionMismatch(""))
    if minimum(rx) < 1 || maximum(rx) > length(x) || minimum(ry) < 1 || maximum(ry) > length(y)
        throw(BoundsError())
    end
    GC.@preserve x y axpy!(length(rx), convert(T, alpha),
                           pointer(x)+(first(rx)-1)*sizeof(T), step(rx),
                           pointer(y)+(first(ry)-1)*sizeof(T), step(ry))
    y
end

function axpby!(n::Integer,
                alpha::T,
                dx::CuArray{T},
                incx::Integer,
                beta::T,
                dy::CuArray{T},
                incy::Integer) where T <: CublasFloat
            scal!(n, beta, dy, incy)
            axpy!(n, alpha, dx, incx, dy, incy)
            dy
end

function axpby!(alpha::Ta,
                x::CuArray{T},
                rx::Union{UnitRange{Ti},AbstractRange{Ti}},
                beta::Tb,
                y::CuArray{T},
                ry::Union{UnitRange{Ti},AbstractRange{Ti}}) where {T<:CublasFloat,Ta<:Number,Tb<:Number,Ti<:Integer}
    length(rx)==length(ry) || throw(DimensionMismatch(""))
    if minimum(rx) < 1 || maximum(rx) > length(x) || minimum(ry) < 1 || maximum(ry) > length(y)
        throw(BoundsError())
    end
    GC.@preserve x y axpby!(length(rx), convert(T, alpha),
                            pointer(x)+(first(rx)-1)*sizeof(T), step(rx), convert(T, beta),
                            pointer(y)+(first(ry)-1)*sizeof(T), step(ry))
    y
end

## iamax
# TODO: fix iamax in julia base
for (fname, elty) in ((:cublasIdamax_v2,:Float64),
                      (:cublasIsamax_v2,:Float32),
                      (:cublasIzamax_v2,:ComplexF64),
                      (:cublasIcamax_v2,:ComplexF32))
    @eval begin
        function iamax(n::Integer,
                       dx::CuArray{$elty},
                       incx::Integer)
            result = Ref{Cint}()
            $fname(handle(), n, dx, incx, result)
            return result[]
        end
    end
end
iamax(dx::CuArray) = iamax(length(dx), dx, 1)

## iamin
# iamin is not in standard blas is a CUBLAS extension
for (fname, elty) in ((:cublasIdamin_v2,:Float64),
                      (:cublasIsamin_v2,:Float32),
                      (:cublasIzamin_v2,:ComplexF64),
                      (:cublasIcamin_v2,:ComplexF32))
    @eval begin
        function iamin(n::Integer,
                       dx::CuArray{$elty},
                       incx::Integer)
            result = Ref{Cint}()
            $fname(handle(), n, dx, incx, result)
            return result[]
        end
    end
end
iamin(dx::CuArray) = iamin(length(dx), dx, 1)

# Level 2
## mv
### gemv
for (fname, elty) in ((:cublasDgemv_v2,:Float64),
                      (:cublasSgemv_v2,:Float32),
                      (:cublasZgemv_v2,:ComplexF64),
                      (:cublasCgemv_v2,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDgemv(
        #   cublasHandle_t handle, cublasOperation_t trans,
        #   int m, int n,
        #   const double *alpha,
        #   const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta,
        #   double *y, int incy)
        function gemv!(trans::Char,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       X::CuVector{$elty},
                       beta::($elty),
                       Y::CuVector{$elty})
            # handle trans
            cutrans = cublasop(trans)
            m,n = size(A)
            # check dimensions
            length(X) == (trans == 'N' ? n : m) && length(Y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            incy = stride(Y,1)
            $fname(handle(), cutrans, m, n, Ref(alpha), A, lda, X, incx, Ref(beta), Y, incy)
            Y
        end
        function gemv(trans::Char, alpha::($elty), A::CuMatrix{$elty}, X::CuVector{$elty})
            gemv!(trans, alpha, A, X, zero($elty), similar(X, $elty, size(A, (trans == 'N' ? 1 : 2))))
        end
        function gemv(trans::Char, A::CuMatrix{$elty}, X::CuVector{$elty})
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
        # cublasStatus_t cublasDgbmv(
        #   cublasHandle_t handle, cublasOperation_t trans,
        #   int m, int n, int kl, int ku,
        #   const double *alpha, const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta, double *y, int incy)
        function gbmv!(trans::Char,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::($elty),
                       y::CuVector{$elty})
            # handle trans
            cutrans = cublasop(trans)
            n = size(A,2)
            # check dimensions
            length(x) == (trans == 'N' ? n : m) && length(y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), cutrans, m, n, kl, ku, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
            y
        end
        function gbmv(trans::Char,
                      m::Integer,
                      kl::Integer,
                      ku::Integer,
                      alpha::($elty),
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
        # cublasStatus_t cublasDsymv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, const double *alpha, const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta, double *y, int incy)
        function symv!(uplo::Char,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::($elty),
                       y::CuVector{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), cuuplo, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
            y
        end
        function symv(uplo::Char, alpha::($elty), A::CuMatrix{$elty}, x::CuVector{$elty})
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
        # cublasStatus_t cublasChemv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, const cuComplex *alpha, const cuComplex *A, int lda,
        #   const cuComplex *x, int incx,
        #   const cuComplex *beta, cuComplex *y, int incy)
        function hemv!(uplo::Char,
                       alpha::$elty,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::$elty,
                       y::CuVector{$elty})
            # TODO: fix dimension check bug in julia
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), cuuplo, n, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
            y
        end
        function hemv(uplo::Char, alpha::($elty), A::CuMatrix{$elty},
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
        # cublasStatus_t cublasDsbmv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, int k, const double *alpha, const double *A, int lda,
        #   const double *x, int incx,
        #   const double *beta, double *y, int incy)
        function sbmv!(uplo::Char,
                       k::Integer,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::($elty),
                       y::CuVector{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            #if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), cuuplo, n, k, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
            y
        end
        function sbmv(uplo::Char, k::Integer, alpha::($elty),
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
        # cublasStatus_t cublasChbmv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
        #   const cuComplex *x, int incx,
        #   const cuComplex *beta, cuComplex *y, int incy)
        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       x::CuVector{$elty},
                       beta::($elty),
                       y::CuVector{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            $fname(handle(), cuuplo, n, k, Ref(alpha), A, lda, x, incx, Ref(beta), y, incy)
            y
        end
        function hbmv(uplo::Char, k::Integer, alpha::($elty),
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
        # cublasStatus_t cublasDtbmv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int n, int k, const double *alpha, const double *A, int lda,
        #   const double *x, int incx)
        function tbmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            cuuplo  = cublasfill(uplo)
            cutrans = cublasop(trans)
            cudiag  = cublasdiag(diag)
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), cuuplo, cutrans, cudiag, n, k, A, lda, x, incx)
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
        # cublasStatus_t cublasDtbsv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int n, int k, const double *alpha, const double *A, int lda,
        #   const double *x, int incx)
        function tbsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::CuMatrix{$elty},
                       x::CuVector{$elty})
            cuuplo  = cublasfill(uplo)
            cutrans = cublasop(trans)
            cudiag  = cublasdiag(diag)
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), cuuplo, cutrans, cudiag, n, k, A, lda, x, incx)
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
        # cublasStatus_t cublasDtrmv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int n, const double *A, int lda,
        #   double *x, int incx)
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
            cuuplo = cublasfill(uplo)
            cutrans = cublasop(trans)
            cudiag = cublasdiag(diag)
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), cuuplo, cutrans, cudiag, n, A, lda, x, incx)
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
        # cublasStatus_t cublasDtrsv(
        #   cublasHandle_t handle, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int n, const double *A, int lda,
        #   double *x, int incx)
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
            cuuplo = cublasfill(uplo)
            cutrans = cublasop(trans)
            cudiag = cublasdiag(diag)
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            $fname(handle(), cuuplo, cutrans, cudiag, n, A, lda, x, incx)
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
        # cublasStatus_t cublasDger(
        #   cublasHandle_t handle, int m, int n, const double *alpha,
        #   const double *x, int incx,
        #   const double *y, int incy,
        #   double *A, int lda)
        function ger!(alpha::$elty,
                      x::CuVector{$elty},
                      y::CuVector{$elty},
                      A::CuMatrix{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            $fname(handle(), m, n, [alpha], x, incx, y, incy, A, lda)
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
        # cublasStatus_t cublasDsyr(
        #   cublasHandle_t handle, cublasFillMode_t uplo, int n,
        #   const double *alpha, const double *x, int incx,
        #   double *A, int lda)
        function syr!(uplo::Char,
                      alpha::$elty,
                      x::CuVector{$elty},
                      A::CuMatrix{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            $fname(handle(), cuuplo, n, [alpha], x, incx, A, lda)
            A
        end
    end
end

### her
for (fname, elty) in ((:cublasZher_v2,:ComplexF64),
                      (:cublasCher_v2,:ComplexF32))
    @eval begin
        function her!(uplo::Char,
                      alpha::$elty,
                      x::CuVector{$elty},
                      A::CuMatrix{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            $fname(handle(), cuuplo, n, [alpha], x, incx, A, lda)
            A
        end
    end
end

### her2
for (fname, elty) in ((:cublasZher2_v2,:ComplexF64),
                      (:cublasCher2_v2,:ComplexF32))
    @eval begin
        function her2!(uplo::Char,
                      alpha::$elty,
                      x::CuVector{$elty},
                      y::CuVector{$elty},
                      A::CuMatrix{$elty})
            cuuplo = cublasfill(uplo)
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            length(y) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            $fname(handle(), cuuplo, n, [alpha], x, incx, y, incy, A, lda)
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
        # cublasStatus_t cublasDgemm(
        #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        #   int m, int n, int k,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb, const double *beta,
        #   double *C, int ldc)
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::($elty),
                       A::CuVecOrMat{$elty},
                       B::CuVecOrMat{$elty},
                       beta::($elty),
                       C::CuVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(handle(), cutransA,cutransB, m, n, k, [alpha], A, lda, B, ldb, [beta],C, ldc)
            C
        end
        function gemm(transA::Char,
                      transB::Char,
                      alpha::($elty),
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            gemm!(transA, transB, alpha, A, B, zero($elty),
                  similar(B, $elty, (size(A, transA == 'N' ? 1 : 2),
                                     size(B, transB == 'N' ? 2 : 1))))
        end
        function gemm(transA::Char,
                      transB::Char,
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            gemm(transA, transB, one($elty), A, B)
        end
    end
end

# GemmEx, with tensor cores
function gemmEx!(transA::Char,
                 transB::Char,
                 alpha::Number,
               A::CuVecOrMat,
               B::CuVecOrMat,
               beta::Number,
               C::CuVecOrMat; algo::cublasGemmAlgo_t=CUBLAS_GEMM_DEFAULT)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch(""))
    end
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    computeType = cudaDataType(eltype(C))
    Atype = cudaDataType(eltype(A))
    Btype = cudaDataType(eltype(B))
    Ctype = cudaDataType(eltype(C))
    cublasGemmEx(handle(), cutransA,cutransB, m, n, k, [convert(eltype(C), alpha)], A, Atype, lda, B, Btype, ldb, [convert(eltype(C), beta)], C, Ctype, ldc, computeType, algo)
    C
end

# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::Vector{<:CuArray{T}}) where {T}
    ptrs = pointer.(batch)
    return CuArray(ptrs)
end

# create a batch of pointers in device memory from a strided device array
@inline function unsafe_strided_batch(strided::CuArray{T}) where {T}
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
        # cublasStatus_t cublasDgemmBatched(
        #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        #   int m, int n, int k,
        #   const double *alpha, const double **A, int lda,
        #   const double **B, int ldb, const double *beta,
        #   double **C, int ldc, int batchCount)
        function gemm_batched!(transA::Char,
                               transB::Char,
                               alpha::($elty),
                               A::Vector{<:CuMatrix{$elty}},
                               B::Vector{<:CuMatrix{$elty}},
                               beta::($elty),
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
            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A[1],2))
            ldb = max(1,stride(B[1],2))
            ldc = max(1,stride(C[1],2))
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            Cptrs = unsafe_batch(C)
            $fname(handle(), cutransA,cutransB, m, n, k, [alpha], Aptrs, lda, Bptrs,
                   ldb, [beta], Cptrs, ldc, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            C
        end
        function gemm_batched(transA::Char,
                      transB::Char,
                      alpha::($elty),
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

        # cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
        #                                   cublasOperation_t transa,
        #                                   cublasOperation_t transb,
        #                                   int m, int n, int k,
        #                                   const double          *alpha,
        #                                   const double          *A, int lda,
        #                                   long long int          strideA,
        #                                   const double          *B, int ldb,
        #                                   long long int          strideB,
        #                                   const double          *beta,
        #                                   double                *C, int ldc,
        #                                   long long int          strideC,
        #                                   int batchCount)

        function gemm_strided_batched!(transA::Char,
                               transB::Char,
                               alpha::($elty),
                               A::CuArray{$elty, 3},
                               B::CuArray{$elty, 3},
                               beta::($elty),
                               C::CuArray{$elty, 3})
           m = size(A, transA == 'N' ? 1 : 2)
           k = size(A, transA == 'N' ? 2 : 1)
           n = size(B, transB == 'N' ? 2 : 1)

           @assert size(A, 3) == size(B, 3) == size(C, 3) "Batch size mismatch"

           if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
               throw(DimensionMismatch(""))
           end
           cutransA = cublasop(transA)
           cutransB = cublasop(transB)
           lda = max(1,stride(A,2))
           ldb = max(1,stride(B,2))
           ldc = max(1,stride(C,2))

           strideA = stride(A, 3)
           strideB = stride(B, 3)
           strideC = stride(C, 3)
           batchCount = size(A, 3)
           $fname(handle(), cutransA,cutransB, m, n, k, [alpha], A, lda, strideA, B,
                  ldb, strideB, [beta], C, ldc, strideC, batchCount)
           C
        end
        function gemm_strided_batched(transA::Char,
                      transB::Char,
                      alpha::($elty),
                      A::CuArray{$elty, 3},
                      B::CuArray{$elty, 3})
            C = similar(B, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1), size(A, 3)))
            gemm_strided_batched!(transA, transB, alpha, A, B, zero($elty), C )
        end
        function gemm_strided_batched(transA::Char,
                      transB::Char,
                      A::CuArray{$elty, 3},
                      B::CuArray{$elty, 3})
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
        # cublasStatus_t cublasDsymm(
        #   cublasHandle_t handle, cublasSideMode_t side,
        #   cublasFillMode_t uplo, int m, int n,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb,
        #   const double *beta, double *C, int ldc)
        function symm!(side::Char,
                       uplo::Char,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty},
                       beta::($elty),
                       C::CuMatrix{$elty})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
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
            $fname(handle(), cuside, cuuplo, m, n, [alpha], A, lda, B, ldb,
                   [beta], C, ldc)
            C
        end
        function symm(side::Char,
                      uplo::Char,
                      alpha::($elty),
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
       # cublasStatus_t cublasDsyrk(
       #   cublasHandle_t handle, cublasFillMode_t uplo,
       #   cublasOperation_t trans, int n, int k,
       #   const double *alpha, const double *A, int lda,
       #   const double *beta, double *C, int ldc)
       function syrk!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::CuVecOrMat{$elty},
                      beta::($elty),
                      C::CuMatrix{$elty})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("syrk!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldc = max(1,stride(C,2))
           $fname(handle(), cuuplo, cutrans, n, k, [alpha], A, lda, [beta], C, ldc)
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
    syrk!(uplo, trans, convert(T,alpha), A, zero(T), similar(A, T, (n, n)))
end
syrk(uplo::Char, trans::Char, A::CuVecOrMat) = syrk(uplo, trans,
                                                              one(eltype(A)),
                                                              A)

for (fname, elty) in ((:cublasDsyrkx,:Float64),
                      (:cublasSsyrkx,:Float32),
                      (:cublasZsyrkx,:ComplexF64),
                      (:cublasCsyrkx,:ComplexF32))
   @eval begin
       # cublasStatus_t cublasDsyrk(
       #   cublasHandle_t handle, cublasFillMode_t uplo,
       #   cublasOperation_t trans, int n, int k,
       #   const double *alpha, const double *A, int lda,
       #   const double *beta, double *C, int ldc)
       function syrkx!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::CuVecOrMat{$elty},
                      B::CuVecOrMat{$elty},
                      beta::($elty),
                      C::CuMatrix{$elty})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("syrkx!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldb = max(1,stride(B,2))
           ldc = max(1,stride(C,2))
           $fname(handle(), cuuplo, cutrans, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc)
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
       # cublasStatus_t cublasChemm(
       #   cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
       #   int m, int n,
       #   const cuComplex *alpha,
       #   const cuComplex *A, int lda,
       #   const cuComplex *B, int ldb,
       #   const cuComplex *beta,
       #   cuComplex *C, int ldc)
       function hemm!(side::Char,
                      uplo::Char,
                      alpha::($elty),
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty},
                      beta::($elty),
                      C::CuMatrix{$elty})
           cuside = cublasside(side)
           cuuplo = cublasfill(uplo)
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
           $fname(handle(), cuside, cuuplo, m, n, [alpha], A, lda, B, ldb, [beta], C, ldc)
           C
       end
       function hemm(uplo::Char,
                     trans::Char,
                     alpha::($elty),
                     A::CuMatrix{$elty},
                     B::CuMatrix{$elty})
           m,n = size(B)
           hemm!( uplo, trans, alpha, A, B, zero($elty), similar(B, $elty, (m,n) ) )
       end
       hemm( uplo::Char, trans::Char, A::CuMatrix{$elty}, B::CuMatrix{$elty}) = hemm( uplo, trans, one($elty), A, B)
    end
end

## herk
for (fname, elty) in ((:cublasZherk_v2,:ComplexF64),
                      (:cublasCherk_v2,:ComplexF32))
   @eval begin
       # cublasStatus_t cublasCherk(
       #   cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
       #   int n, int k,
       #   const float *alpha, const cuComplex *A, int lda,
       #   const float *beta, cuComplex *C, int ldc)
       function herk!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::CuVecOrMat{$elty},
                      beta::($elty),
                      C::CuMatrix{$elty})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("herk!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldc = max(1,stride(C,2))
           $fname(handle(), cuuplo, cutrans, n, k, [alpha], A, lda, [beta], C, ldc)
           C
       end
       function herk(uplo::Char, trans::Char, alpha::($elty), A::CuVecOrMat{$elty})
           n = size(A, trans == 'N' ? 1 : 2)
           herk!(uplo, trans, alpha, A, zero($elty), similar(A, $elty, (n,n)))
       end
       herk(uplo::Char, trans::Char, A::CuVecOrMat{$elty}) = herk(uplo, trans, one($elty), A)
   end
end

## syr2k
for (fname, elty) in ((:cublasDsyr2k_v2,:Float64),
                      (:cublasSsyr2k_v2,:Float32),
                      (:cublasZsyr2k_v2,:ComplexF64),
                      (:cublasCsyr2k_v2,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDsyr2k(
        #   cublasHandle_t handle,
        #   cublasFillMode_t uplo, cublasOperation_t trans,
        #   int n, int k,
        #   const double *alpha,
        #   const double *A, int lda,
        #   const double *B, int ldb,
        #   const double *beta,
        #   double *C, int ldc)
        function syr2k!(uplo::Char,
                        trans::Char,
                        alpha::($elty),
                        A::CuVecOrMat{$elty},
                        B::CuVecOrMat{$elty},
                        beta::($elty),
                        C::CuMatrix{$elty})
            # TODO: check size of B in julia (syr2k!)
            cuuplo = cublasfill(uplo)
            cutrans = cublasop(trans)
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
            $fname(handle(), cuuplo, cutrans, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc)
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
syr2k(uplo::Char, trans::Char, A::CuVecOrMat, B::CuVecOrMat) = syr2k(uplo, trans, one(eltype(A)), A, B)

## her2k
for (fname, elty1, elty2) in ((:cublasZher2k_v2,:ComplexF64,:Float64),
                              (:cublasCher2k_v2,:ComplexF32,:Float32))
   @eval begin
       # cublasStatus_t cublasZher2k(
       #   cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
       #   int n, int k,
       #   const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
       #   const cuDoubleComplex *B, int ldb,
       #   const double *beta, cuDoubleComplex *C, int ldc)
       function her2k!(uplo::Char,
                       trans::Char,
                       alpha::($elty1),
                       A::CuVecOrMat{$elty1},
                       B::CuVecOrMat{$elty1},
                       beta::($elty2),
                       C::CuMatrix{$elty1})
           # TODO: check size of B in julia (her2k!)
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
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
           $fname(handle(), cuuplo, cutrans, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc)
           C
       end
       function her2k(uplo::Char,
                      trans::Char,
                      alpha::($elty1),
                      A::CuVecOrMat{$elty1},
                      B::CuVecOrMat{$elty1})
           n = size(A, trans == 'N' ? 1 : 2)
           her2k!(uplo, trans, alpha, A, B, zero($elty2), similar(A, $elty1, (n,n)))
       end
       her2k(uplo::Char,
             trans::Char,
             A::CuVecOrMat{$elty1},
             B::CuVecOrMat{$elty1}) = her2k(uplo, trans, one($elty1), A, B)
   end
end

## (TR) Triangular matrix and vector multiplication and solution
for (mmname, smname, elty) in
        ((:cublasDtrmm_v2,:cublasDtrsm_v2,:Float64),
         (:cublasStrmm_v2,:cublasStrsm_v2,:Float32),
         (:cublasZtrmm_v2,:cublasZtrsm_v2,:ComplexF64),
         (:cublasCtrmm_v2,:cublasCtrsm_v2,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDtrmm(cublasHandle_t handle,
        #   cublasSideMode_t side, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int m, int n,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb,
        #   double *C, int ldc)
        # Note: CUBLAS differs from BLAS API for trmm
        #   BLAS: inplace modification of B
        #   CUBLAS: store result in C
        function trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty},
                       C::CuMatrix{$elty})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
            cutransa = cublasop(transa)
            cudiag = cublasdiag(diag)
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
            $mmname(handle(), cuside, cuuplo, cutransa, cudiag, m, n, [alpha], A, lda, B, ldb, C, ldc)
            C
        end
        function trmm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::($elty),
                      A::CuMatrix{$elty},
                      B::CuMatrix{$elty})
            trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
        end
        # cublasStatus_t cublasDtrsm(cublasHandle_t handle,
        #   cublasSideMode_t side, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int m, int n,
        #   const double *alpha,
        #   const double *A, int lda,
        #   double *B, int ldb)
        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::($elty),
                       A::CuMatrix{$elty},
                       B::CuMatrix{$elty})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
            cutransa = cublasop(transa)
            cudiag = cublasdiag(diag)
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            $smname(handle(), cuside, cuuplo, cutransa, cudiag, m, n, [alpha], A, lda, B, ldb)
            B
        end
        function trsm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::($elty),
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
        # cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
        #   cublasSideMode_t side, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int m, int n,
        #   const double *alpha,
        #   const double **A, int lda,
        #   double **B, int ldb,
        #   int batchCount)
        function trsm_batched!(side::Char,
                               uplo::Char,
                               transa::Char,
                               diag::Char,
                               alpha::($elty),
                               A::Vector{<:CuMatrix{$elty}},
                               B::Vector{<:CuMatrix{$elty}})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
            cutransa = cublasop(transa)
            cudiag = cublasdiag(diag)
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
            $fname(handle(), cuside, cuuplo, cutransa, cudiag, m, n, [alpha], Aptrs, lda, Bptrs, ldb, length(A))
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            B
        end
        function trsm_batched(side::Char,
                              uplo::Char,
                              transa::Char,
                              diag::Char,
                              alpha::($elty),
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
       # cublasStatus_t cublasCgeam(
       #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
       #   int m, int n,
       #   const cuComplex *alpha,
       #   const cuComplex *A, int lda,
       #   const cuComplex *B, int ldb,
       #   const cuComplex *beta,
       #   cuComplex *C, int ldc)
       function geam!(transa::Char,
                      transb::Char,
                      alpha::($elty),
                      A::CuMatrix{$elty},
                      beta::($elty),
                      B::CuMatrix{$elty},
                      C::CuMatrix{$elty})
           cutransa = cublasop(transa)
           cutransb = cublasop(transb)
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
           $fname(handle(), cutransa, cutransb, m, n, [alpha], A, lda, [beta], B, ldb, C, ldc)
           C
       end
       function geam(transa::Char,
                     transb::Char,
                     alpha::($elty),
                     A::CuMatrix{$elty},
                     beta::($elty),
                     B::CuMatrix{$elty})
           m,n = size(B)
           if ((transb == 'T' || transb == 'C'))
               geam!( transa, transb, alpha, A, beta, B, similar(B, $elty, (n,m) ) )
           end
           if (transb == 'N')
               geam!( transa, transb, alpha, A, beta, B, similar(B, $elty, (m,n) ) )
           end
       end
       geam( uplo::Char, trans::Char, A::CuMatrix{$elty}, B::CuMatrix{$elty}) = geam( uplo, trans, one($elty), A, one($elty), B)
    end
end

## getrfBatched - performs LU factorizations

for (fname, elty) in
        ((:cublasDgetrfBatched,:Float64),
         (:cublasSgetrfBatched,:Float32),
         (:cublasZgetrfBatched,:ComplexF64),
         (:cublasCgetrfBatched,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDgetrfBatched(
        #   cublasHandle_t handle, int n, double **A,
        #   int lda, int *PivotArray, int *infoArray,
        #   int batchSize)
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
function getrf_strided_batched!(A::CuArray{<:Any, 3}, pivot::Bool)
    m,n = size(A,1), size(A,2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    lda = max(1,stride(A,2))

    Aptrs = unsafe_strided_batch(A)
    return getrf_batched!(n, Aptrs, lda, pivot)..., A
end
getrf_strided_batched(A::CuArray{<:Any, 3}, pivot::Bool) = getrf_strided_batched!(copy(A), pivot)


## getriBatched - performs batched matrix inversion

for (fname, elty) in
        ((:cublasDgetriBatched,:Float64),
         (:cublasSgetriBatched,:Float32),
         (:cublasZgetriBatched,:ComplexF64),
         (:cublasCgetriBatched,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDgetriBatched(
        #   cublasHandle_t handle, int n, double **A,
        #   int lda, int *PivotArray, double **C,
        #   int ldc, int *info, int batchSize)
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
        # cublasStatus_t cublasDmatinvBatched(
        #   cublasHandle_t handle, int n, double **A,
        #   int lda, double **C, int ldc,
        #   int *info, int batchSize)
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
        # cublasStatus_t cublasDgeqrfBatched(
        #   cublasHandle_t handle, int n, int m,
        #   double **A, int lda, double **TauArray,
        #   int *infoArray, int batchSize)
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
        # cublasStatus_t cublasDgelsBatched(
        #   cublasHandle_t handle, int m, int n,
        #   int nrhs, double **A, int lda,
        #   double **C, int ldc, int *infoArray,
        #   int *devInfoArray, int batchSize)
        function gels_batched!(trans::Char,
                              A::Vector{<:CuMatrix{$elty}},
                              C::Vector{<:CuMatrix{$elty}})
            cutrans = cublasop(trans)
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
            $fname(handle(), cutrans, m, n, nrhs, Aptrs, lda, Cptrs, ldc, [info], infoarray, length(A))
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
       # cublasStatus_t cublasCdgmm(
       #   cublasHandle_t handle, cublasSideMode_t mode,
       #   int m, int n,
       #   const cuComplex *A, int lda,
       #   const cuComplex *X, int incx,
       #   cuComplex *C, int ldc)
       function dgmm!(mode::Char,
                      A::CuMatrix{$elty},
                      X::CuVector{$elty},
                      C::CuMatrix{$elty})
           cuside = cublasside(mode)
           m, n = size(C)
           mA, nA = size(A)
           lx = length(X)
           if ((mA != m) || (nA != n )) throw(DimensionMismatch("")) end
           if ((mode == 'L') && (lx != m)) throw(DimensionMismatch("")) end
           if ((mode == 'R') && (lx != n)) throw(DimensionMismatch("")) end
           lda = max(1,stride(A,2))
           incx = stride(X,1)
           ldc = max(1,stride(C,2))
           $fname(handle(), cuside, m, n, A, lda, X, incx, C, ldc)
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
for (fname, elty) in
        ((:cublasXtSgemm,:Float32),
         (:cublasXtDgemm,:Float64),
         (:cublasXtCgemm,:ComplexF32),
         (:cublasXtZgemm,:ComplexF64))
    @eval begin
        # cublasStatus_t cublasDgemm(
        #   cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        #   int m, int n, int k,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb, const double *beta,
        #   double *C, int ldc)
        function xt_gemm!(transA::Char,
                       transB::Char,
                       alpha::($elty),
                       A::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                       B::Union{CuVecOrMat{$elty}, VecOrMat{$elty}},
                       beta::($elty),
                       C::Union{CuVecOrMat{$elty}, VecOrMat{$elty}})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            cutransA = cublasop(transA)
            cutransB = cublasop(transB)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            $fname(xt_handle(), cutransA,cutransB, m, n, k, [alpha], A, lda, B, ldb, [beta],C, ldc)
            C
        end
        function xt_gemm(transA::Char,
                      transB::Char,
                      alpha::($elty),
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
       # cublasStatus_t cublasChemm(
       #   cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
       #   int m, int n,
       #   const cuComplex *alpha,
       #   const cuComplex *A, int lda,
       #   const cuComplex *B, int ldb,
       #   const cuComplex *beta,
       #   cuComplex *C, int ldc)
       function xt_hemm!(side::Char,
                      uplo::Char,
                      alpha::($elty),
                      A::Union{Matrix{$elty}, CuMatrix{$elty}},
                      B::Union{Matrix{$elty}, CuMatrix{$elty}},
                      beta::($elty),
                      C::Union{Matrix{$elty}, CuMatrix{$elty}})
           cuside = cublasside(side)
           cuuplo = cublasfill(uplo)
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
           $fname(xt_handle(), cuside, cuuplo, m, n, [alpha], A, lda, B, ldb, [beta], C, ldc)
           C
       end
       function xt_hemm(uplo::Char,
                     trans::Char,
                     alpha::($elty),
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
        # cublasStatus_t cublasDsymm(
        #   cublasHandle_t handle, cublasSideMode_t side,
        #   cublasFillMode_t uplo, int m, int n,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb,
        #   const double *beta, double *C, int ldc)
        function xt_symm!(side::Char,
                       uplo::Char,
                       alpha::($elty),
                       A::Union{Matrix{$elty}, CuMatrix{$elty}},
                       B::Union{Matrix{$elty}, CuMatrix{$elty}},
                       beta::($elty),
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
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
            $fname(xt_handle(), cuside, cuuplo, m, n, [alpha], A, lda, B, ldb,
                   [beta], C, ldc)
            C
        end
        function xt_symm(side::Char,
                      uplo::Char,
                      alpha::($elty),
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
       # cublasStatus_t cublasDsyrk(
       #   cublasHandle_t handle, cublasFillMode_t uplo,
       #   cublasOperation_t trans, int n, int k,
       #   const double *alpha, const double *A, int lda,
       #   const double *beta, double *C, int ldc)
       function xt_syrk!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                      beta::($elty),
                      C::Union{Matrix{$elty}, CuMatrix{$elty}})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("syrk!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldc = max(1,stride(C,2))
           $fname(xt_handle(), cuuplo, cutrans, n, k, [alpha], A, lda, [beta], C, ldc)
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
    xt_syrk!(uplo, trans, convert(T,alpha), A, zero(T), similar(A, T, (n, n)))
end
xt_syrk(uplo::Char, trans::Char, A::Union{VecOrMat, CuVecOrMat}) = xt_syrk(uplo, trans, one(eltype(A)), A)

for (fname, elty) in ((:cublasXtDsyrkx,:Float64),
                      (:cublasXtSsyrkx,:Float32),
                      (:cublasXtZsyrkx,:ComplexF64),
                      (:cublasXtCsyrkx,:ComplexF32))
   @eval begin
       # cublasStatus_t cublasDsyrk(
       #   cublasHandle_t handle, cublasFillMode_t uplo,
       #   cublasOperation_t trans, int n, int k,
       #   const double *alpha, const double *A, int lda,
       #   const double *beta, double *C, int ldc)
       function xt_syrkx!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                      B::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                      beta::($elty),
                      C::Union{Matrix{$elty}, CuMatrix{$elty}})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("xt_syrkx!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldb = max(1,stride(B,2))
           ldc = max(1,stride(C,2))
           $fname(xt_handle(), cuuplo, cutrans, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc)
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
    xt_syrkx!(uplo, trans, convert(T,alpha), A, B, convert(T,beta), similar(A, T, (n, n)))
end
xt_syrkx(uplo::Char, trans::Char, A::Union{VecOrMat, CuVecOrMat}, B::Union{VecOrMat, CuVecOrMat}) = xt_syrkx(uplo, trans,
                                                                 one(eltype(A)), A, B,
                                                                 zero(eltype(B)))

for (fname, elty) in ((:cublasXtZherk,:ComplexF64),
                      (:cublasXtCherk,:ComplexF32))
   @eval begin
       # cublasStatus_t cublasCherk(
       #   cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
       #   int n, int k,
       #   const float *alpha, const cuComplex *A, int lda,
       #   const float *beta, cuComplex *C, int ldc)
       function xt_herk!(uplo::Char,
                      trans::Char,
                      alpha::($elty),
                      A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}},
                      beta::($elty),
                      C::Union{Matrix{$elty}, CuMatrix{$elty}})
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
           mC, n = size(C)
           if mC != n throw(DimensionMismatch("C must be square")) end
           nn = size(A, trans == 'N' ? 1 : 2)
           if nn != n throw(DimensionMismatch("herk!")) end
           k  = size(A, trans == 'N' ? 2 : 1)
           lda = max(1,stride(A,2))
           ldc = max(1,stride(C,2))
           $fname(xt_handle(), cuuplo, cutrans, n, k, [alpha], A, lda, [beta], C, ldc)
           C
       end
       function xt_herk(uplo::Char, trans::Char, alpha::($elty), A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}})
           n = size(A, trans == 'N' ? 1 : 2)
           xt_herk!(uplo, trans, alpha, A, zero($elty), similar(A, $elty, (n,n)))
       end
       xt_herk(uplo::Char, trans::Char, A::Union{VecOrMat{$elty}, CuVecOrMat{$elty}}) = xt_herk(uplo, trans, one($elty), A)
   end
end

for (fname, elty1, elty2) in ((:cublasXtZher2k,:ComplexF64,:Float64),
                              (:cublasXtCher2k,:ComplexF32,:Float32))
   @eval begin
       # cublasStatus_t cublasZher2k(
       #   cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
       #   int n, int k,
       #   const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
       #   const cuDoubleComplex *B, int ldb,
       #   const double *beta, cuDoubleComplex *C, int ldc)
       function xt_her2k!(uplo::Char,
                       trans::Char,
                       alpha::($elty1),
                       A::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}},
                       B::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}},
                       beta::($elty2),
                       C::Union{Matrix{$elty1}, CuMatrix{$elty1}})
           # TODO: check size of B in julia (her2k!)
           cuuplo = cublasfill(uplo)
           cutrans = cublasop(trans)
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
           $fname(xt_handle(), cuuplo, cutrans, n, k, [alpha], A, lda, B, ldb, [beta], C, ldc)
           C
       end
       function xt_her2k(uplo::Char,
                      trans::Char,
                      alpha::($elty1),
                      A::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}},
                      B::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}})
           n = size(A, trans == 'N' ? 1 : 2)
           xt_her2k!(uplo, trans, alpha, A, B, zero($elty2), similar(A, $elty1, (n,n)))
       end
       xt_her2k(uplo::Char,
             trans::Char,
             A::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}},
             B::Union{VecOrMat{$elty1}, CuVecOrMat{$elty1}}) = xt_her2k(uplo, trans, one($elty1), A, B)
   end
end

for (mmname, smname, elty) in
        ((:cublasXtDtrmm,:cublasXtDtrsm,:Float64),
         (:cublasXtStrmm,:cublasXtStrsm,:Float32),
         (:cublasXtZtrmm,:cublasXtZtrsm,:ComplexF64),
         (:cublasXtCtrmm,:cublasXtCtrsm,:ComplexF32))
    @eval begin
        # cublasStatus_t cublasDtrmm(cublasHandle_t handle,
        #   cublasSideMode_t side, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int m, int n,
        #   const double *alpha, const double *A, int lda,
        #   const double *B, int ldb,
        #   double *C, int ldc)
        # Note: CUBLAS differs from BLAS API for trmm
        #   BLAS: inplace modification of B
        #   CUBLAS: store result in C
        function xt_trmm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::($elty),
                       A::Union{Matrix{$elty}, CuMatrix{$elty}},
                       B::Union{Matrix{$elty}, CuMatrix{$elty}},
                       C::Union{Matrix{$elty}, CuMatrix{$elty}})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
            cutransa = cublasop(transa)
            cudiag = cublasdiag(diag)
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
            $mmname(xt_handle(), cuside, cuuplo, cutransa, cudiag, m, n, [alpha], A, lda, B, ldb, C, ldc)
            C
        end
        function xt_trmm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::($elty),
                      A::Union{CuMatrix{$elty}, Matrix{$elty}},
                      B::Union{CuMatrix{$elty}, Matrix{$elty}})
            xt_trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
        end
        # cublasStatus_t cublasDtrsm(cublasHandle_t handle,
        #   cublasSideMode_t side, cublasFillMode_t uplo,
        #   cublasOperation_t trans, cublasDiagType_t diag,
        #   int m, int n,
        #   const double *alpha,
        #   const double *A, int lda,
        #   double *B, int ldb)
        function xt_trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::($elty),
                       A::Union{CuMatrix{$elty}, Matrix{$elty}},
                       B::Union{CuMatrix{$elty}, Matrix{$elty}})
            cuside = cublasside(side)
            cuuplo = cublasfill(uplo)
            cutransa = cublasop(transa)
            cudiag = cublasdiag(diag)
            m, n = size(B)
            mA, nA = size(A)
            # TODO: clean up error messages
            if mA != nA throw(DimensionMismatch("A must be square")) end
            if nA != (side == 'L' ? m : n) throw(DimensionMismatch("trsm!")) end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            $smname(xt_handle(), cuside, cuuplo, cutransa, cudiag, m, n, [alpha], A, lda, B, ldb)
            B
        end
        function xt_trsm(side::Char,
                      uplo::Char,
                      transa::Char,
                      diag::Char,
                      alpha::($elty),
                      A::Union{CuMatrix{$elty}, Matrix{$elty}},
                      B::Union{CuMatrix{$elty}, Matrix{$elty}})
            xt_trsm!(side, uplo, transa, diag, alpha, A, copy(B))
        end
    end
end

# high-level functionality
#
# modeled from julia/src/base/linalg/blas.jl
# originally authored by Nick Henderson <nwh@stanford.edu> (2014-08-26, MIT licensed)

function cublasCreate()
  handle_ref = Ref{cublasHandle_t}()
  cublasCreate_v2(handle_ref)
  handle_ref[]
end

function cublasGetVersion(handle)
  version = Ref{Cint}()
  cublasGetVersion_v2(handle, version)
  major, ver = divrem(version[], 10000)
  minor, patch = divrem(ver, 100)
  VersionNumber(major, minor, patch)
end

function cublasXtCreate()
  handle_ref = Ref{cublasXtHandle_t}()
  cublasXtCreate(handle_ref)
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

version() = VersionNumber(cublasGetProperty(CUDA.MAJOR_VERSION),
                          cublasGetProperty(CUDA.MINOR_VERSION),
                          cublasGetProperty(CUDA.PATCH_LEVEL))

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

# most level 1 routines are intended for use on vectors, so only accept a single stride.
# however, it is often convenient to also use these routines on arbitrary arrays,
# interpreting them as vectors. this does not work with arbitrary strides, so we
# define a union matching arrays with only a non-unit stride in the first dimension.
const StridedCuVecOrDenseMat{T} = Union{StridedCuVector{T}, DenseCuArray{T}}

## copy
for (fname, fname_64, elty) in ((:cublasDcopy_v2, :cublasDcopy_v2_64, :Float64),
                                (:cublasScopy_v2, :cublasScopy_v2_64, :Float32),
                                (:cublasZcopy_v2, :cublasZcopy_v2_64, :ComplexF64),
                                (:cublasCcopy_v2, :cublasCcopy_v2_64, :ComplexF32))
    @eval begin
        function copy!(n::Integer,
                       x::StridedCuVecOrDenseMat{$elty},
                       y::StridedCuVecOrDenseMat{$elty},)
            if CUBLAS.version() >= v"12.0"
              $fname_64(handle(), n, x, stride(x, 1), y, stride(y, 1))
            else
              $fname(handle(), n, x, stride(x, 1), y, stride(y, 1))
            end
            y
        end
    end
end
function copy!(n::Integer, x::StridedCuVecOrDenseMat{T},
               y::StridedCuVecOrDenseMat{T}) where {T <: Union{Float16, ComplexF16}}
    copyto!(y, x) # bad
end

## scal
for (fname, fname_64, elty) in ((:cublasDscal_v2, :cublasDscal_v2_64, :Float64),
                                (:cublasSscal_v2, :cublasSscal_v2_64, :Float32),
                                (:cublasZscal_v2, :cublasZscal_v2_64, :ComplexF64),
                                (:cublasCscal_v2, :cublasCscal_v2_64, :ComplexF32))
    @eval begin
        function scal!(n::Integer,
                       alpha::Number,
                       x::StridedCuVecOrDenseMat{$elty})
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, alpha, x, stride(x, 1))
            else
                $fname(handle(), n, alpha, x, stride(x, 1))
            end
            x
        end
    end
end
function scal!(n::Integer, alpha::Number, x::StridedCuVecOrDenseMat{Float16})
    α = convert(Float32, alpha)
    cublasScalEx(handle(), n, Ref{Float32}(α), Float32, x, Float16, stride(x, 1), Float32)
    return x
end
# specific variants in case x is complex and alpha is real
for (fname, fname_64, elty, celty) in ((:cublasCsscal_v2, :cublasCsscal_v2_64, :Float32, :ComplexF32),
                                       (:cublasZdscal_v2, :cublasZdscal_v2_64, :Float64, :ComplexF64))
    @eval begin
        function scal!(n::Integer,
                       alpha::$elty,
                       x::StridedCuVecOrDenseMat{$celty})
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, alpha, x, stride(x, 1))
            else
                $fname(handle(), n, alpha, x, stride(x, 1))
            end
            x
        end
    end
end
function scal!(n::Integer, alpha::Number, x::StridedCuVecOrDenseMat{ComplexF16})
    wide_x = widen.(x)
    scal!(n, alpha, wide_x)
    thin_x = convert(typeof(x), wide_x)
    copyto!(x, thin_x)
    return x
end

## dot, dotc, dotu
for (jname, fname, fname_64, elty) in ((:dot, :cublasDdot_v2, :cublasDdot_v2_64, :Float64),
                                       (:dot, :cublasSdot_v2, :cublasSdot_v2_64, :Float32),
                                       (:dotc, :cublasZdotc_v2, :cublasZdotc_v2_64, :ComplexF64),
                                       (:dotc, :cublasCdotc_v2, :cublasCdotc_v2_64, :ComplexF32),
                                       (:dotu, :cublasZdotu_v2, :cublasZdotu_v2_64, :ComplexF64),
                                       (:dotu, :cublasCdotu_v2, :cublasCdotu_v2_64, :ComplexF32))
    @eval begin
        function $jname(n::Integer,
                        x::StridedCuVecOrDenseMat{$elty},
                        y::StridedCuVecOrDenseMat{$elty})
            result = Ref{$elty}()
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, x, stride(x, 1), y, stride(y, 1), result)
            else
                $fname(handle(), n, x, stride(x, 1), y, stride(y, 1), result)
            end
            return result[]
        end
    end
end
function dot(n::Integer, x::StridedCuVecOrDenseMat{Float16}, y::StridedCuVecOrDenseMat{Float16})
    result = Ref{Float16}()
    cublasDotEx(handle(), n, x, Float16, stride(x, 1), y, Float16, stride(y, 1), result, Float16, Float32)
    return result[]
end
function dotc(n::Integer, x::StridedCuVecOrDenseMat{ComplexF16}, y::StridedCuVecOrDenseMat{ComplexF16})
    convert(ComplexF16, dotc(n, convert(CuArray{ComplexF32}, x), convert(CuArray{ComplexF32}, y)))
end
function dotu(n::Integer, x::StridedCuVecOrDenseMat{ComplexF16}, y::StridedCuVecOrDenseMat{ComplexF16})
    convert(ComplexF16, dotu(n, convert(CuArray{ComplexF32}, x), convert(CuArray{ComplexF32}, y)))
end

## nrm2
for (fname, fname_64, elty, ret_type) in ((:cublasDnrm2_v2, :cublasDnrm2_v2_64, :Float64, :Float64),
                                          (:cublasSnrm2_v2, :cublasSnrm2_v2_64, :Float32, :Float32),
                                          (:cublasDznrm2_v2, :cublasDznrm2_v2_64, :ComplexF64, :Float64),
                                          (:cublasScnrm2_v2, :cublasScnrm2_v2_64, :ComplexF32, :Float32))
    @eval begin
        function nrm2(n::Integer,
                      X::StridedCuVecOrDenseMat{$elty})
            result = Ref{$ret_type}()
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, X, stride(X, 1), result)
            else
                $fname(handle(), n, X, stride(X, 1), result)
            end
            return result[]
        end
    end
end
nrm2(x::StridedCuVecOrDenseMat) = nrm2(length(x), x)

function nrm2(n::Integer, x::StridedCuVecOrDenseMat{Float16})
    result = Ref{Float16}()
    cublasNrm2Ex(handle(), n, x, Float16, stride(x, 1), result, Float16, Float32)
    return result[]
end
function nrm2(n::Integer, x::StridedCuVecOrDenseMat{ComplexF16})
    wide_x = widen.(x)
    nrm    = nrm2(n, wide_x)
    return convert(Float16, nrm)
end

## asum
for (fname, fname_64, elty, ret_type) in ((:cublasDasum_v2, :cublasDasum_v2_64, :Float64, :Float64),
                                          (:cublasSasum_v2, :cublasSasum_v2_64, :Float32, :Float32),
                                          (:cublasDzasum_v2, :cublasDzasum_v2_64, :ComplexF64, :Float64),
                                          (:cublasScasum_v2, :cublasScasum_v2_64, :ComplexF32, :Float32))
    @eval begin
        function asum(n::Integer,
                      x::StridedCuVecOrDenseMat{$elty})
            result = Ref{$ret_type}()
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, x, stride(x, 1), result)
            else
                $fname(handle(), n, x, stride(x, 1), result)
            end
            return result[]
        end
    end
end

## axpy
for (fname, fname_64, elty) in ((:cublasDaxpy_v2, :cublasDaxpy_v2_64, :Float64),
                                (:cublasSaxpy_v2, :cublasSaxpy_v2_64, :Float32),
                                (:cublasZaxpy_v2, :cublasZaxpy_v2_64, :ComplexF64),
                                (:cublasCaxpy_v2, :cublasCaxpy_v2_64, :ComplexF32))
    @eval begin
        function axpy!(n::Integer,
                       alpha::Number,
                       dx::StridedCuVecOrDenseMat{$elty},
                       dy::StridedCuVecOrDenseMat{$elty})
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, alpha, dx, stride(dx, 1), dy, stride(dy, 1))
            else
                $fname(handle(), n, alpha, dx, stride(dx, 1), dy, stride(dy, 1))
            end
            dy
        end
    end
end

function axpy!(n::Integer, alpha::Number, dx::StridedCuVecOrDenseMat{Float16}, dy::StridedCuVecOrDenseMat{Float16})
    α = convert(Float32, alpha)
    cublasAxpyEx(handle(), n, Ref{Float32}(α), Float32, dx, Float16, stride(dx, 1), dy, Float16, stride(dy, 1), Float32)
    return dy
end
function axpy!(n::Integer, alpha::Number, dx::StridedCuVecOrDenseMat{ComplexF16}, dy::StridedCuVecOrDenseMat{ComplexF16})
    wide_x = widen.(dx)
    wide_y = widen.(dy)
    axpy!(n, alpha, wide_x, wide_y)
    thin_y = convert(typeof(dy), wide_y)
    copyto!(dy, thin_y)
    return dy
end

## rot
for (fname, fname_64, elty, sty) in ((:cublasSrot_v2, :cublasSrot_v2_64, :Float32, :Number),
                                     (:cublasDrot_v2, :cublasDrot_v2_64, :Float64, :Number),
                                     (:cublasCrot_v2, :cublasCrot_v2_64, :ComplexF32, :Number),
                                     (:cublasCsrot_v2, :cublasCsrot_v2_64, :ComplexF32, :Real),
                                     (:cublasZrot_v2, :cublasZrot_v2_64, :ComplexF64, :Number),
                                     (:cublasZdrot_v2, :cublasZdrot_v2_64, :ComplexF64, :Real))
    @eval begin
        function rot!(n::Integer,
                      x::StridedCuVecOrDenseMat{$elty},
                      y::StridedCuVecOrDenseMat{$elty},
                      c::Real,
                      s::$sty)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), n, x, stride(x, 1), y, stride(y, 1), c, s)
            else
                $fname(handle(), n, x, stride(x, 1), y, stride(y, 1), c, s)
            end
            x, y
        end
    end
end

## swap
for (fname, fname_64, elty) in ((:cublasSswap_v2, :cublasSswap_v2_64, :Float32),
                                (:cublasDswap_v2, :cublasDswap_v2_64, :Float64),
                                (:cublasCswap_v2, :cublasCswap_v2_64, :ComplexF32),
                                (:cublasZswap_v2, :cublasZswap_v2_64, :ComplexF64))
    @eval begin
        function swap!(n::Integer,
                       x::StridedCuVecOrDenseMat{$elty},
                       y::StridedCuVecOrDenseMat{$elty})
            if CUBLAS.version() >= v"12.0"
               $fname_64(handle(), n, x, stride(x, 1), y, stride(y, 1))
            else
                $fname(handle(), n, x, stride(x, 1), y, stride(y, 1))
            end
            x, y
        end
    end
end

function axpby!(n::Integer,
                alpha::Number,
                dx::StridedCuVecOrDenseMat{T},
                beta::Number,
                dy::StridedCuVecOrDenseMat{T}) where T <: Union{Float16, ComplexF16, CublasFloat}
            scal!(n, beta, dy)
            axpy!(n, alpha, dx, dy)
            dy
end

## iamax
# TODO: fix iamax in julia base
for (fname, fname_64, elty) in ((:cublasIdamax_v2, :cublasIdamax_v2_64, :Float64),
                                (:cublasIsamax_v2, :cublasIsamax_v2_64, :Float32),
                                (:cublasIzamax_v2, :cublasIzamax_v2_64, :ComplexF64),
                                (:cublasIcamax_v2, :cublasIcamax_v2_64, :ComplexF32))
    @eval begin
        function iamax(n::Integer,
                       dx::StridedCuVecOrDenseMat{$elty})
            if CUBLAS.version() >= v"12.0"
                result = Ref{Int64}()
                $fname_64(handle(), n, dx, stride(dx, 1), result)
            else
                result = Ref{Cint}()
                $fname(handle(), n, dx, stride(dx, 1), result)
            end
            return result[]
        end
    end
end
iamax(dx::StridedCuVecOrDenseMat) = iamax(length(dx), dx)

## iamin
# iamin is not in standard blas is a CUBLAS extension
for (fname, fname_64, elty) in ((:cublasIdamin_v2, :cublasIdamin_v2_64, :Float64),
                                (:cublasIsamin_v2, :cublasIsamin_v2_64, :Float32),
                                (:cublasIzamin_v2, :cublasIzamin_v2_64, :ComplexF64),
                                (:cublasIcamin_v2, :cublasIcamin_v2_64, :ComplexF32))
    @eval begin
        function iamin(n::Integer,
                       dx::StridedCuVecOrDenseMat{$elty},)
            if CUBLAS.version() >= v"12.0"
                result = Ref{Int64}()
                $fname_64(handle(), n, dx, stride(dx, 1), result)
            else
                result = Ref{Cint}()
                $fname(handle(), n, dx, stride(dx, 1), result)
            end
            return result[]
        end
    end
end
iamin(dx::StridedCuVecOrDenseMat) = iamin(length(dx), dx)

# Level 2
## mv
### gemv
for (fname, fname_64, elty) in ((:cublasDgemv_v2, :cublasDgemv_v2_64, :Float64),
                                (:cublasSgemv_v2, :cublasSgemv_v2_64, :Float32),
                                (:cublasZgemv_v2, :cublasZgemv_v2_64, :ComplexF64),
                                (:cublasCgemv_v2, :cublasCgemv_v2_64, :ComplexF32))
    @eval begin
        function gemv!(trans::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            # handle trans
            m,n = size(A)
            # check dimensions
            length(x) == (trans == 'N' ? n : m) && length(y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function gemv(trans::Char, alpha::Number,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    gemv!(trans, alpha, A, x, zero(T), similar(x, size(A, (trans == 'N' ? 1 : 2))))
end
function gemv(trans::Char, A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    gemv!(trans, one(T), A, x, zero(T), similar(x, T, size(A, (trans == 'N' ? 1 : 2))))
end

for (fname, fname_64, eltyin, eltyout) in (
    (:cublasDgemvBatched, :cublasDgemvBatched_64, :Float64, :Float64),
    (:cublasSgemvBatched, :cublasSgemvBatched_64, :Float32, :Float32),
    (:cublasHSHgemvBatched, :cublasHSHgemvBatched, :Float16, :Float16),
    (:cublasHSSgemvBatched, :cublasHSSgemvBatched, :Float16, :Float32),
    (:cublasZgemvBatched, :cublasZgemvBatched_64, :ComplexF64, :ComplexF64),
    (:cublasCgemvBatched, :cublasCgemvBatched_64, :ComplexF32, :ComplexF32))
    @eval begin
        function gemv_batched!(trans::Char,
                            alpha::Number,
                            A::Vector{<:StridedCuMatrix{$eltyin}},
                            x::Vector{<:StridedCuVector{$eltyin}},
                            beta::Number,
                            y::Vector{<:StridedCuVector{$eltyout}})
            if length(A) != length(x) || length(A) != length(y)
                throw(DimensionMismatch("Lengths of inputs must be the same"))
            end
            m = size(A[1], 1)
            n = size(A[1], 2)
            for (i, (As,xs,ys)) in enumerate(zip(A,x,y))
                if size(As) != (m, n)
                    throw(DimensionMismatch("A[$i] has different dimension from A[1]. Dimensions between A's should be identical."))
                end
                if length(xs) != (trans == 'N' ? n : m) || length(ys) != (trans == 'N' ? m : n)
                    throw(DimensionMismatch("Input $i: A has dimension $(size(As)), x has dimension $(size(xs)), y has dimension $(size(ys))"))
                end
            end
            lda = max(1,stride(A[1],2))
            incx = stride(x[1],1)
            incy = stride(y[1],1)
            Aptrs = unsafe_batch(A)
            xptrs = unsafe_batch(x)
            yptrs = unsafe_batch(y)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), trans, m, n, alpha, Aptrs, lda, xptrs, incx, beta, yptrs, incy, length(A))
            else
                $fname(handle(), trans, m, n, alpha, Aptrs, lda, xptrs, incx, beta, yptrs, incy, length(A))
            end
            unsafe_free!(yptrs)
            unsafe_free!(xptrs)
            unsafe_free!(Aptrs)

            y
        end
    end
end

for (fname, fname_64, eltyin, eltyout) in (
    (:cublasDgemvStridedBatched, :cublasDgemvStridedBatched_64, :Float64, :Float64),
    (:cublasSgemvStridedBatched, :cublasSgemvStridedBatched_64, :Float32, :Float32),
    (:cublasHSHgemvStridedBatched, :cublasHSHgemvStridedBatched, :Float16, :Float16),
    (:cublasHSSgemvStridedBatched, :cublasHSSgemvStridedBatched, :Float16, :Float32),
    (:cublasZgemvStridedBatched, :cublasZgemvStridedBatched_64, :ComplexF64, :ComplexF64),
    (:cublasCgemvStridedBatched, :cublasCgemvStridedBatched_64, :ComplexF32, :ComplexF32))
    @eval begin
        function gemv_strided_batched!(trans::Char,
                            alpha::Number,
                            A::AbstractArray{$eltyin, 3},
                            x::AbstractArray{$eltyin, 2},
                            beta::Number,
                            y::AbstractArray{$eltyout, 2})
            if size(A, 3) != size(x, 2) || size(A, 3) != size(y, 2)
                throw(DimensionMismatch("Batch sizes must be equal for all inputs"))
            end
            m = size(A, 1)
            n = size(A, 2)
            if size(y, 1) != (trans == 'N' ? m : n) || size(x, 1) != (trans == 'N' ? n : m)
                throw(DimensionMismatch("A has dimension $(size(A)), x has dimension $(size(x)), y has dimension $(size(y))"))
            end

            lda = max(1,stride(A, 2))
            incx = stride(x,1)
            incy = stride(y,1)
            strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
            stridex = size(x, 2) == 1 ? 0 : stride(x, 2)
            stridey = stride(y, 2)
            batchCount = size(A, 3)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
            else
                $fname(handle(), trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount)
            end
            y
        end
    end
end

### (GB) general banded matrix-vector multiplication
for (fname, fname_64, elty) in ((:cublasDgbmv_v2, :cublasDgbmv_v2_64, :Float64),
                                (:cublasSgbmv_v2, :cublasSgbmv_v2_64, :Float32),
                                (:cublasZgbmv_v2, :cublasZgbmv_v2_64, :ComplexF64),
                                (:cublasCgbmv_v2, :cublasCgbmv_v2_64, :ComplexF32))
    @eval begin
        function gbmv!(trans::Char,
                       m::Integer,
                       kl::Integer,
                       ku::Integer,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            n = size(A,2)
            # check dimensions
            length(x) == (trans == 'N' ? n : m) && length(y) == (trans == 'N' ? m : n) || throw(DimensionMismatch(""))
            # compute increments
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function gbmv(trans::Char, m::Integer, kl::Integer, ku::Integer, alpha::Number,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    # TODO: fix gbmv bug in julia
    n = size(A,2)
    leny = trans == 'N' ? m : n
    gbmv!(trans, m, kl, ku, alpha, A, x, zero(T), similar(x, leny))
end
function gbmv(trans::Char, m::Integer, kl::Integer, ku::Integer,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    gbmv(trans, m, kl, ku, one(T), A, x)
end

### spmv
for (fname, fname_64, elty) in ((:cublasDspmv_v2, :cublasDspmv_v2_64, :Float64),
                                (:cublasSspmv_v2, :cublasSspmv_v2_64, :Float32))
    @eval begin
        function spmv!(uplo::Char,
                       alpha::Number,
                       AP::StridedCuVector{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            n = round(Int, (sqrt(8*length(AP))-1)/2)
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, AP, x, incx, beta, y, incy)
            else
                $fname(handle(), uplo, n, alpha, AP, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function spmv(uplo::Char, alpha::Number,
              AP::StridedCuVector{T}, x::StridedCuVector{T}) where T
    spmv!(uplo, alpha, AP, x, zero(T), similar(x))
end
function spmv(uplo::Char, AP::StridedCuVector{T}, x::StridedCuVector{T}) where T
    spmv(uplo, one(T), AP, x)
end

### symv
for (fname, fname_64, elty) in ((:cublasDsymv_v2, :cublasDsymv_v2_64, :Float64),
                                (:cublasSsymv_v2, :cublasSsymv_v2_64, :Float32),
                                (:cublasZsymv_v2, :cublasZsymv_v2_64, :ComplexF64),
                                (:cublasCsymv_v2, :cublasCsymv_v2_64, :ComplexF32))
    # Note that the complex symv are not BLAS but auiliary functions in LAPACK
    @eval begin
        function symv!(uplo::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function symv(uplo::Char, alpha::Number,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
        symv!(uplo, alpha, A, x, zero(T), similar(x))
end
function symv(uplo::Char, A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    symv(uplo, one(T), A, x)
end

### hemv
# TODO: fix chemv_ function call bug in julia
for (fname, fname_64, elty) in ((:cublasZhemv_v2, :cublasZhemv_v2_64, :ComplexF64),
                                (:cublasChemv_v2, :cublasChemv_v2_64, :ComplexF32))
    @eval begin
        function hemv!(uplo::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            # TODO: fix dimension check bug in julia
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if m != length(x) || m != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), uplo, n, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function hemv(uplo::Char, alpha::Number, A::StridedCuMatrix{T},
              x::StridedCuVector{T}) where T
    hemv!(uplo, alpha, A, x, zero(T), similar(x))
end
function hemv(uplo::Char, A::StridedCuMatrix{T},
              x::StridedCuVector{T}) where T
    hemv(uplo, one(T), A, x)
end

### sbmv, (SB) symmetric banded matrix-vector multiplication
# cublas only has this for D and S
# TODO: check in julia, blas may not have sbmv for C and Z!
for (fname, fname_64, elty) in ((:cublasDsbmv_v2, :cublasDsbmv_v2_64, :Float64),
                                (:cublasSsbmv_v2, :cublasSsbmv_v2_64, :Float32))
    @eval begin
        function sbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            m, n = size(A)
            #if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function sbmv(uplo::Char, k::Integer, alpha::Number,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    n = size(A,2)
    sbmv!(uplo, k, alpha, A, x, zero(T), similar(x, n))
end
function sbmv(uplo::Char, k::Integer, A::StridedCuMatrix{T},
              x::StridedCuVector{T}) where T
    sbmv(uplo, k, one(T), A, x)
end

### hbmv, (HB) Hermitian banded matrix-vector multiplication
for (fname, fname_64, elty) in ((:cublasZhbmv_v2, :cublasZhbmv_v2_64, :ComplexF64),
                                (:cublasChbmv_v2, :cublasChbmv_v2_64, :ComplexF32))
    @eval begin
        function hbmv!(uplo::Char,
                       k::Integer,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty},
                       beta::Number,
                       y::StridedCuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) || n != length(y) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            incy = stride(y,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            else
                $fname(handle(), uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
            end
            y
        end
    end
end
function hbmv(uplo::Char, k::Integer, alpha::Number,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    n = size(A,2)
    hbmv!(uplo, k, alpha, A, x, zero(T), similar(x, n))
end
function hbmv(uplo::Char, k::Integer, A::StridedCuMatrix{T},
              x::StridedCuVector{T}) where T
    hbmv(uplo, k, one(T), A, x)
end

### tbmv, (TB) triangular banded matrix-vector multiplication
for (fname, fname_64, elty) in ((:cublasStbmv_v2, :cublasStbmv_v2_64, :Float32),
                                (:cublasDtbmv_v2, :cublasDtbmv_v2_64, :Float64),
                                (:cublasZtbmv_v2, :cublasZtbmv_v2_64, :ComplexF64),
                                (:cublasCtbmv_v2, :cublasCtbmv_v2_64, :ComplexF32))
    @eval begin
        function tbmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            else
                $fname(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            end
            x
        end
    end
end
function tbmv(uplo::Char, trans::Char, diag::Char, k::Integer,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    tbmv!(uplo, trans, diag, k, A, copy(x))
end

### tbsv, (TB) triangular banded matrix solve
for (fname, fname_64, elty) in ((:cublasStbsv_v2, :cublasStbsv_v2_64, :Float32),
                                (:cublasDtbsv_v2, :cublasDtbsv_v2_64, :Float64),
                                (:cublasZtbsv_v2, :cublasZtbsv_v2_64, :ComplexF64),
                                (:cublasCtbsv_v2, :cublasCtbsv_v2_64, :ComplexF32))
    @eval begin
        function tbsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       k::Integer,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty})
            m, n = size(A)
            if !(1<=(1+k)<=n) throw(DimensionMismatch("Incorrect number of bands")) end
            if m < 1+k throw(DimensionMismatch("Array A has fewer than 1+k rows")) end
            if n != length(x) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            else
                $fname(handle(), uplo, trans, diag, n, k, A, lda, x, incx)
            end
            x
        end
    end
end
function tbsv(uplo::Char, trans::Char, diag::Char, k::Integer,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    tbsv!(uplo, trans, diag, k, A, copy(x))
end

### trmv, Triangular matrix-vector multiplication
for (fname, fname_64, elty) in ((:cublasDtrmv_v2, :cublasDtrmv_v2_64, :Float64),
                                (:cublasStrmv_v2, :cublasStrmv_v2_64, :Float32),
                                (:cublasZtrmv_v2, :cublasZtrmv_v2_64, :ComplexF64),
                                (:cublasCtrmv_v2, :cublasCtrmv_v2_64, :ComplexF32))
    @eval begin
        function trmv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, diag, n, A, lda, x, incx)
            else
                $fname(handle(), uplo, trans, diag, n, A, lda, x, incx)
            end
            x
        end
    end
end
function trmv(uplo::Char, trans::Char, diag::Char,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    trmv!(uplo, trans, diag, A, copy(x))
end

### trsv, Triangular matrix-vector solve
for (fname, fname_64, elty) in ((:cublasDtrsv_v2, :cublasDtrsv_v2_64, :Float64),
                                (:cublasStrsv_v2, :cublasStrsv_v2_64, :Float32),
                                (:cublasZtrsv_v2, :cublasZtrsv_v2_64, :ComplexF64),
                                (:cublasCtrsv_v2, :cublasCtrsv_v2_64, :ComplexF32))
    @eval begin
        function trsv!(uplo::Char,
                       trans::Char,
                       diag::Char,
                       A::StridedCuMatrix{$elty},
                       x::StridedCuVector{$elty})
            m, n = size(A)
            if m != n throw(DimensionMismatch("Matrix A is $m by $n but must be square")) end
            if n != length(x)
                throw(DimensionMismatch("length(x)=$(length(x)) does not match size(A)=$(size(A))"))
            end
            lda = max(1,stride(A,2))
            incx = stride(x,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, diag, n, A, lda, x, incx)
            else
                $fname(handle(), uplo, trans, diag, n, A, lda, x, incx)
            end
            x
        end
    end
end
function trsv(uplo::Char, trans::Char, diag::Char,
              A::StridedCuMatrix{T}, x::StridedCuVector{T}) where T
    trsv!(uplo, trans, diag, A, copy(x))
end

### ger
for (fname, fname_64, elty) in ((:cublasDger_v2, :cublasDger_v2_64, :Float64),
                                (:cublasSger_v2, :cublasSger_v2_64, :Float32),
                                (:cublasZgerc_v2, :cublasZgerc_v2_64, :ComplexF64),
                                (:cublasCgerc_v2, :cublasCgerc_v2_64, :ComplexF32))
    @eval begin
        function ger!(alpha::Number,
                      x::StridedCuVector{$elty},
                      y::StridedCuVector{$elty},
                      A::StridedCuMatrix{$elty})
            m, n = size(A)
            m == length(x) || throw(DimensionMismatch(""))
            n == length(y) || throw(DimensionMismatch(""))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), m, n, alpha, x, incx, y, incy, A, lda)
            else
                $fname(handle(), m, n, alpha, x, incx, y, incy, A, lda)
            end
            A
        end
    end
end

### spr
for (fname, fname_64, elty) in ((:cublasDspr_v2, :cublasDspr_v2_64, :Float64),
                                (:cublasSspr_v2, :cublasSspr_v2_64, :Float32))
    @eval begin
        function spr!(uplo::Char,
                      alpha::Number,
                      x::StridedCuVector{$elty},
                      AP::StridedCuVector{$elty})
            n = round(Int, (sqrt(8*length(AP))-1)/2)
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, x, incx, AP)
            else
                $fname(handle(), uplo, n, alpha, x, incx, AP)
            end
            AP
        end
    end
end

### syr
# TODO: check calls in julia b/c blas may not define syr for Z and C
for (fname, fname_64, elty) in ((:cublasDsyr_v2, :cublasDsyr_v2_64, :Float64),
                                (:cublasSsyr_v2, :cublasSsyr_v2_64, :Float32),
                                (:cublasZsyr_v2, :cublasZsyr_v2_64, :ComplexF64),
                                (:cublasCsyr_v2, :cublasCsyr_v2_64, :ComplexF32))
    @eval begin
        function syr!(uplo::Char,
                      alpha::Number,
                      x::StridedCuVector{$elty},
                      A::StridedCuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, x, incx, A, lda)
            else
                $fname(handle(), uplo, n, alpha, x, incx, A, lda)
            end
            A
        end
    end
end

### her
for (fname, fname_64, elty) in ((:cublasZher_v2, :cublasZher_v2_64, :ComplexF64),
                                (:cublasCher_v2, :cublasCher_v2_64, :ComplexF32))
    @eval begin
        function her!(uplo::Char,
                      alpha::Number,
                      x::StridedCuVector{$elty},
                      A::StridedCuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            lda = max(1,stride(A,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, x, incx, A, lda)
            else
                $fname(handle(), uplo, n, alpha, x, incx, A, lda)
            end
            A
        end
    end
end

### her2
for (fname, fname_64, elty) in ((:cublasZher2_v2, :cublasZher2_v2_64, :ComplexF64),
                                (:cublasCher2_v2, :cublasCher2_v2_64, :ComplexF32))
    @eval begin
        function her2!(uplo::Char,
                      alpha::Number,
                      x::StridedCuVector{$elty},
                      y::StridedCuVector{$elty},
                      A::StridedCuMatrix{$elty})
            m, n = size(A)
            m == n || throw(DimensionMismatch("Matrix A is $m by $n but must be square"))
            length(x) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            length(y) == n || throw(DimensionMismatch("Length of vector must be the same as the matrix dimensions"))
            incx = stride(x,1)
            incy = stride(y,1)
            lda = max(1,stride(A,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, n, alpha, x, incx, y, incy, A, lda)
            else
                $fname(handle(), uplo, n, alpha, x, incx, y, incy, A, lda)
            end
            A
        end
    end
end

# Level 3
## (GE) general matrix-matrix multiplication
for (fname, fname_64, elty) in ((:cublasDgemm_v2, :cublasDgemm_v2_64, :Float64),
                                (:cublasSgemm_v2, :cublasSgemm_v2_64, :Float32),
                                (:cublasHgemm, :cublasHgemm, :Float16),
                                (:cublasZgemm_v2, :cublasZgemm_v2_64, :ComplexF64),
                                (:cublasCgemm_v2, :cublasCgemm_v2_64, :ComplexF32))
    @eval begin
        function gemm!(transA::Char,
                       transB::Char,
                       alpha::Number,
                       A::StridedCuVecOrMat{$elty},
                       B::StridedCuVecOrMat{$elty},
                       beta::Number,
                       C::StridedCuVecOrMat{$elty})
            m = size(A, transA == 'N' ? 1 : 2)
            k = size(A, transA == 'N' ? 2 : 1)
            n = size(B, transB == 'N' ? 2 : 1)
            if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
                throw(DimensionMismatch(""))
            end
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
    end
end
function gemm(transA::Char, transB::Char, alpha::Number,
              A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}) where T
    gemm!(transA, transB, alpha, A, B, zero(T),
          similar(B, (size(A, transA == 'N' ? 1 : 2),
                      size(B, transB == 'N' ? 2 : 1))))
end
function gemm(transA::Char, transB::Char,
              A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}) where T
    gemm(transA, transB, one(T), A, B)
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

    # source: CUBLAS Features and Technical Specifications
    if Float16 in sig && cap < v"5.3"
        return nothing
    end

    math_mode = CUDA.math_mode()
    reduced_precision = CUDA.math_precision()

    if sig === (Float16, Float16)
        # NOTE: Float16=Float16*Float16 can also happen in 32-bit compute
        return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_16F_PEDANTIC : CUBLAS_COMPUTE_16F
    end

    if sig === (Int8, Int32)
        # starting with CUDA 11.2, this is unsupported (NVIDIA bug #3221266)
        # TODO: might be fixed in a later version?
        version() >= v"11.3.1" && return nothing

        # Int32=Int8*Int8 requires m,n,k to be multiples of 4
        # https://forums.developer.nvidia.com/t/cublasgemmex-cant-use-cuda-r-8i-compute-type-on-gtx1080/58100/2
        if m%4 == 0 && n%4 == 0 && k%4 == 0
            return math_mode==CUDA.PEDANTIC_MATH ? CUBLAS_COMPUTE_32I_PEDANTIC : CUBLAS_COMPUTE_32I
        end
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
                 @nospecialize(A::StridedCuVecOrMat),
                 @nospecialize(B::StridedCuVecOrMat),
                 @nospecialize(beta::Number),
                 @nospecialize(C::StridedCuVecOrMat);
                 algo::cublasGemmAlgo_t=CUBLAS_GEMM_DEFAULT)
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("A has dimension $(size(A)), B has dimension $(size(B)) and C has dimension $(size(C))"))
    end
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))
    computeType = gemmExComputeType(eltype(A), eltype(B), eltype(C), m, k, n)
    isnothing(computeType) &&
        throw(ArgumentError("gemmEx does not support $(eltype(C))=$(eltype(A))*$(eltype(B))"))
    computeT = juliaStorageType(eltype(C), computeType)
    if version() >= v"11.0"
        # with CUDA 11, the compute type encodes the math mode.
        cublasGemmEx(handle(), transA, transB, m, n, k, Ref{computeT}(alpha), A, eltype(A), lda, B,
                    eltype(B), ldb, Ref{computeT}(beta), C, eltype(C), ldc, computeType, algo)
    else
        # before CUDA 11, it was a plain cudaDataType.
        computeType = convert(cudaDataType, computeT)
        cublasGemmEx_old(handle(), transA, transB, m, n, k, Ref{computeT}(alpha), A, eltype(A), lda, B,
                    eltype(B), ldb, Ref{computeT}(beta), C, eltype(C), ldc, computeType, algo)
    end
    C
end

function gemmBatchedEx!(transA::Char, transB::Char,
                 @nospecialize(alpha::Number),
                 @nospecialize(A::Vector{<:StridedCuVecOrMat}),
                 @nospecialize(B::Vector{<:StridedCuVecOrMat}),
                 @nospecialize(beta::Number),
                 @nospecialize(C::Vector{<:StridedCuVecOrMat});
                 algo::cublasGemmAlgo_t=CUBLAS_GEMM_DEFAULT)
    if length(A) != length(B) || length(A) != length(C)
        throw(DimensionMismatch("Lengths of inputs must be the same"))
    end
    for (i, (As,Bs,Cs)) in enumerate(zip(A,B,C))
        m = size(As, transA == 'N' ? 1 : 2)
        k = size(As, transA == 'N' ? 2 : 1)
        n = size(Bs, transB == 'N' ? 2 : 1)
        if m != size(Cs,1) || n != size(Cs,2) || k != size(Bs, transB == 'N' ? 1 : 2)
            throw(DimensionMismatch("Input $i: A has dimension $(size(As)), B has dimension $(size(Bs)), C has dimension $(size(Cs))"))
        end
    end
    m = size(A[1], transA == 'N' ? 1 : 2)
    k = size(A[1], transA == 'N' ? 2 : 1)
    n = size(B[1], transB == 'N' ? 2 : 1)
    lda = max(1,stride(A[1],2))
    ldb = max(1,stride(B[1],2))
    ldc = max(1,stride(C[1],2))
    computeType = gemmExComputeType(eltype(A[1]), eltype(B[1]), eltype(C[1]), m, k, n)
    isnothing(computeType) &&
    throw(ArgumentError("gemmEx does not support $(eltype(C))=$(eltype(A))*$(eltype(B))"))
    computeT = juliaStorageType(eltype(C[1]), computeType)
    Aptrs = unsafe_batch(A)
    Bptrs = unsafe_batch(B)
    Cptrs = unsafe_batch(C)
    if version() >= v"11.0"
        # with CUDA 11, the compute type encodes the math mode.
        cublasGemmBatchedEx(handle(), transA, transB, m, n, k, Ref{computeT}(alpha), Aptrs, eltype(A[1]), lda, Bptrs,
                            eltype(B[1]), ldb, Ref{computeT}(beta), Cptrs, eltype(C[1]), ldc, length(A), computeType, algo)
    else
        error("Not implemented for CUDA 11 and below.")
    end
    unsafe_free!(Cptrs)
    unsafe_free!(Bptrs)
    unsafe_free!(Aptrs)

    C
end

function gemmStridedBatchedEx!(transA::Char, transB::Char,
                 @nospecialize(alpha::Number),
                 @nospecialize(A::AbstractArray{Ta, 3}),
                 @nospecialize(B::AbstractArray{Tb, 3}),
                 @nospecialize(beta::Number),
                 @nospecialize(C::AbstractArray{Tc, 3});
                 algo::cublasGemmAlgo_t=CUBLAS_GEMM_DEFAULT) where {Ta, Tb, Tc}
    if size(A, 3) != size(B, 3) || size(A, 3) != size(C, 3)
        throw(DimensionMismatch("Batch sizes must be equal for all inputs"))
    end
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, transB == 'N' ? 2 : 1)
    if m != size(C,1) || n != size(C,2) || k != size(B, transB == 'N' ? 1 : 2)
        throw(DimensionMismatch("A has dimension $(size(A)), B has dimension $(size(B)), C has dimension $(size(C))"))
    end
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    ldc = max(1,stride(C,2))

    strideA = size(A, 3) == 1 ? 0 : stride(A, 3)
    strideB = size(B, 3) == 1 ? 0 : stride(B, 3)
    strideC = stride(C, 3)
    batchCount = size(C, 3)

    computeType = gemmExComputeType(eltype(A), eltype(B), eltype(C), m, k, n)
    isnothing(computeType) &&
    throw(ArgumentError("gemmEx does not support $(eltype(C))=$(eltype(A))*$(eltype(B))"))
    computeT = juliaStorageType(eltype(C), computeType)
    if version() >= v"11.0"
        # with CUDA 11, the compute type encodes the math mode.
        cublasGemmStridedBatchedEx(handle(), transA, transB, m, n, k, Ref{computeT}(alpha), A, eltype(A), lda, strideA,
                                   B, eltype(B), ldb, strideB, Ref{computeT}(beta), C, eltype(C), ldc, strideC,
                                   batchCount, computeType, algo)
    else
        error("Not implemented for CUDA 11 and below.")
    end
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

## (GE) general matrix-matrix multiplication grouped batched
for (fname, fname_64, elty) in ((:cublasSgemmGroupedBatched, :cublasSgemmGroupedBatched_64, :Float32),
                                (:cublasDgemmGroupedBatched, :cublasDgemmGroupedBatched_64, :Float64))
    @eval begin
        function gemm_grouped_batched!(transA::Vector{Char},
                                       transB::Vector{Char},
                                       alpha::Vector{$elty},
                                       A::Vector{<:Vector{<:StridedCuMatrix{$elty}}},
                                       B::Vector{<:Vector{<:StridedCuMatrix{$elty}}},
                                       beta::Vector{$elty},
                                       C::Vector{<:Vector{<:StridedCuMatrix{$elty}}})

            if length(A) != length(B) || length(A) != length(C)
                throw(DimensionMismatch("A, B and C must contain the same number of groups"))
            end
            group_count = length(A)
            for i=1:group_count
                if length(A[i]) != length(B[i]) || length(A[i]) != length(C[i])
                    throw(DimensionMismatch("A, B and C must contain the same number of matrices"))
                end
            end
            group_size = length.(A)

            for i = 1:group_count
                m = size(A[i][1], transA[i] == 'N' ? 1 : 2)
                k = size(A[i][1], transA[i] == 'N' ? 2 : 1)
                n = size(B[i][1], transB[i] == 'N' ? 2 : 1)
                if m != size(C[i][1],1) || n != size(C[i][1],2) || k != size(B[i][1], transB[i] == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end

            transa = convert.(cublasOperation_t, transA)
            transb = convert.(cublasOperation_t, transB)
            m = [size(A[i][1], transA[i] == 'N' ? 1 : 2) for i = 1 : group_count]
            k = [size(A[i][1], transA[i] == 'N' ? 2 : 1) for i = 1 : group_count]
            n = [size(B[i][1], transB[i] == 'N' ? 2 : 1) for i = 1 : group_count]
            lda = [max(1,stride(A[i][1],2)) for i = 1 : group_count]
            ldb = [max(1,stride(B[i][1],2)) for i = 1 : group_count]
            ldc = [max(1,stride(C[i][1],2)) for i = 1 : group_count]
            Aptrs = unsafe_batch(reduce(vcat, A))
            Bptrs = unsafe_batch(reduce(vcat, B))
            Cptrs = unsafe_batch(reduce(vcat, C))

            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), transa, transb, m, n, k, alpha, Aptrs, lda,
                          Bptrs, ldb, beta, Cptrs, ldc, group_count, group_size)
            else
                $fname(handle(), transa, transb, m, n, k, alpha, Aptrs, lda,
                          Bptrs, ldb, beta, Cptrs, ldc, group_count, group_size)
            end
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            C
        end
    end

    # Group size hardcoded to one
    @eval begin
        function gemm_grouped_batched!(transA::Vector{Char},
                                       transB::Vector{Char},
                                       alpha::Vector{$elty},
                                       A::Vector{<:StridedCuMatrix{$elty}},
                                       B::Vector{<:StridedCuMatrix{$elty}},
                                       beta::Vector{$elty},
                                       C::Vector{<:StridedCuMatrix{$elty}})
            if length(A) != length(B) || length(A) != length(C)
                throw(DimensionMismatch("A, B and C must contain the same number of matrices"))
            end

            group_count = length(A)
            group_size = ones(Int64, group_count)

            for i = 1:group_count
                m = size(A[i], transA[i] == 'N' ? 1 : 2)
                k = size(A[i], transA[i] == 'N' ? 2 : 1)
                n = size(B[i], transB[i] == 'N' ? 2 : 1)
                if m != size(C[i],1) || n != size(C[i],2) || k != size(B[i], transB[i] == 'N' ? 1 : 2)
                    throw(DimensionMismatch(""))
                end
            end

            transa = convert.(cublasOperation_t, transA)
            transb = convert.(cublasOperation_t, transB)
            m = [size(A[i], transA[i] == 'N' ? 1 : 2) for i = 1 : group_count]
            k = [size(A[i], transA[i] == 'N' ? 2 : 1) for i = 1 : group_count]
            n = [size(B[i], transB[i] == 'N' ? 2 : 1) for i = 1 : group_count]
            lda = [max(1,stride(A[i],2)) for i = 1 : group_count]
            ldb = [max(1,stride(B[i],2)) for i = 1 : group_count]
            ldc = [max(1,stride(C[i],2)) for i = 1 : group_count]
            Aptrs = unsafe_batch(A)
            Bptrs = unsafe_batch(B)
            Cptrs = unsafe_batch(C)

            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), transa, transb, m, n, k, alpha, Aptrs, lda,
                          Bptrs, ldb, beta, Cptrs, ldc, group_count, group_size)
            else
                $fname(handle(), transa, transb, m, n, k, alpha, Aptrs, lda,
                          Bptrs, ldb, beta, Cptrs, ldc, group_count, group_size)
            end
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)
            C
        end
    end
end

function gemm_grouped_batched(transA::Vector{Char}, transB::Vector{Char}, alpha::Vector{T},
                              A::Vector{<:Vector{<:StridedCuMatrix{T}}}, B::Vector{<:Vector{<:StridedCuMatrix{T}}}) where T
    num_groups = length(A)
    group_sizes = length.(A)
    beta = [zero(T) for i = 1:num_groups]
    C = [[similar(B[i][j], (size(A[i][j], transA[i] == 'N' ? 1 : 2), size(B[i][j], transB[i] == 'N' ? 2 : 1))) for j in 1:group_sizes[i]] for i in 1:num_groups]
    gemm_grouped_batched!(transA, transB, alpha, A, B, beta, C)
end

function gemm_grouped_batched(transA::Vector{Char}, transB::Vector{Char},
                              A::Vector{<:Vector{<:StridedCuMatrix{T}}}, B::Vector{<:Vector{<:StridedCuMatrix{T}}}) where T
    alpha = [one(T) for i = 1:length(transA)]
    gemm_grouped_batched(transA, transB, alpha, A, B)
end

# Group size hardcoded to one
function gemm_grouped_batched(transA::Vector{Char}, transB::Vector{Char}, alpha::Vector{T},
    A::Vector{<:StridedCuMatrix{T}}, B::Vector{<:StridedCuMatrix{T}}) where T
beta = [zero(T) for i = 1:length(transA)]
C = CuMatrix{T}[similar(B[i], (size(A[i], transA[i] == 'N' ? 1 : 2), size(B[i], transB[i] == 'N' ? 2 : 1))) for i in 1:length(A)]
gemm_grouped_batched!(transA, transB, alpha, A, B, beta, C)
end

function gemm_grouped_batched(transA::Vector{Char}, transB::Vector{Char},
    A::Vector{<:StridedCuMatrix{T}}, B::Vector{<:StridedCuMatrix{T}}) where T
alpha = [one(T) for i = 1:length(transA)]
gemm_grouped_batched(transA, transB, alpha, A, B)
end

## (GE) general matrix-matrix multiplication batched
for (fname, fname_64, elty) in ((:cublasDgemmBatched, :cublasDgemmBatched_64, :Float64),
                                (:cublasSgemmBatched, :cublasSgemmBatched_64, :Float32),
                                (:cublasHgemmBatched, :cublasHgemmBatched, :Float16),
                                (:cublasZgemmBatched, :cublasZgemmBatched_64, :ComplexF64),
                                (:cublasCgemmBatched, :cublasCgemmBatched_64, :ComplexF32))
    @eval begin
        function gemm_batched!(transA::Char,
                               transB::Char,
                               alpha::Number,
                               A::Vector{<:StridedCuMatrix{$elty}},
                               B::Vector{<:StridedCuMatrix{$elty}},
                               beta::Number,
                               C::Vector{<:StridedCuMatrix{$elty}})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), transA, transB, m, n, k, alpha, Aptrs, lda, Bptrs,
                          ldb, beta, Cptrs, ldc, length(A))
            else
                $fname(handle(), transA, transB, m, n, k, alpha, Aptrs, lda, Bptrs,
                       ldb, beta, Cptrs, ldc, length(A))
            end
            unsafe_free!(Cptrs)
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            C
        end
    end
end

function gemm_batched(transA::Char, transB::Char, alpha::Number,
                      A::Vector{<:StridedCuMatrix{T}}, B::Vector{<:StridedCuMatrix{T}}) where T
    C = CuMatrix{T}[similar(B[1], (size(A[1], transA == 'N' ? 1 : 2),size(B[1], transB == 'N' ? 2 : 1))) for i in 1:length(A)]
    gemm_batched!(transA, transB, alpha, A, B, zero(T), C )
end
function gemm_batched(transA::Char, transB::Char,
                      A::Vector{<:StridedCuMatrix{T}}, B::Vector{<:StridedCuMatrix{T}}) where T
    gemm_batched(transA, transB, one(T), A, B)
end

## (GE) general matrix-matrix multiplication strided batched
for (fname, fname_64, elty) in ((:cublasDgemmStridedBatched, :cublasDgemmStridedBatched_64, :Float64),
                                (:cublasSgemmStridedBatched, :cublasSgemmStridedBatched_64, :Float32),
                                (:cublasHgemmStridedBatched, :cublasHgemmStridedBatched, :Float16),
                                (:cublasZgemmStridedBatched, :cublasZgemmStridedBatched_64, :ComplexF64),
                                (:cublasCgemmStridedBatched, :cublasCgemmStridedBatched_64, :ComplexF32))
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
           if CUBLAS.version() >= v"12.0"
              $fname_64(handle(), transA, transB, m, n, k, alpha, A, lda, strideA, B,
                     ldb, strideB, beta, C, ldc, strideC, batchCount)
           else
              $fname(handle(), transA, transB, m, n, k, alpha, A, lda, strideA, B,
                     ldb, strideB, beta, C, ldc, strideC, batchCount)
           end
           C
        end
    end
end
function gemm_strided_batched(transA::Char, transB::Char, alpha::Number,
                              A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    C = similar(B, (size(A, transA == 'N' ? 1 : 2),
                    size(B, transB == 'N' ? 2 : 1),
                    max(size(A, 3), size(B, 3))))
    gemm_strided_batched!(transA, transB, alpha, A, B, zero(T), C )
end
function gemm_strided_batched(transA::Char, transB::Char, A::AbstractArray{T, 3},
                              B::AbstractArray{T, 3}) where T
    gemm_strided_batched(transA, transB, one(T), A, B)
end

## (Sy) symmetric matrix-matrix and matrix-vector multiplication
for (fname, fname_64, elty) in ((:cublasDsymm_v2, :cublasDsymm_v2_64, :Float64),
                                (:cublasSsymm_v2, :cublasSsymm_v2_64, :Float32),
                                (:cublasZsymm_v2, :cublasZsymm_v2_64, :ComplexF64),
                                (:cublasCsymm_v2, :cublasCsymm_v2_64, :ComplexF32))
    # TODO: fix julia dimension checks in symm!
    @eval begin
        function symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       B::StridedCuMatrix{$elty},
                       beta::Number,
                       C::StridedCuMatrix{$elty})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
    end
end
function symm(side::Char, uplo::Char, alpha::Number,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    symm!(side, uplo, alpha, A, B, zero(T), similar(B))
end
function symm(side::Char, uplo::Char,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    symm(side, uplo, one(T), A, B)
end

## syrk
for (fname, fname_64, elty) in ((:cublasDsyrk_v2, :cublasDsyrk_v2_64, :Float64),
                                (:cublasSsyrk_v2, :cublasSsyrk_v2_64, :Float32),
                                (:cublasZsyrk_v2, :cublasZsyrk_v2_64, :ComplexF64),
                                (:cublasCsyrk_v2, :cublasCsyrk_v2_64, :ComplexF32))
    @eval begin
        function syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::StridedCuVecOrMat{$elty},
                       beta::Number,
                       C::StridedCuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            else
                $fname(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            end
            C
        end
    end
end
function syrk(uplo::Char, trans::Char, alpha::Number, A::StridedCuVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    syrk!(uplo, trans, alpha, A, zero(T), similar(A, (n, n)))
end
function syrk(uplo::Char, trans::Char, A::StridedCuVecOrMat)
    syrk(uplo, trans, one(eltype(A)), A)
end

for (fname, fname_64, elty) in ((:cublasDsyrkx, :cublasDsyrkx_64, :Float64),
                                (:cublasSsyrkx, :cublasSsyrkx_64, :Float32),
                                (:cublasZsyrkx, :cublasZsyrkx_64, :ComplexF64),
                                (:cublasCsyrkx, :cublasCsyrkx_64, :ComplexF32))
    @eval begin
        function syrkx!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::StridedCuVecOrMat{$elty},
                       B::StridedCuVecOrMat{$elty},
                       beta::Number,
                       C::StridedCuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("syrkx!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldb = max(1,stride(B,2))
            ldc = max(1,stride(C,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
    end
end
function syrkx(uplo::Char, trans::Char, alpha::Number, A::StridedCuVecOrMat{T},
                beta::Number, B::StridedCuVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    syrkx!(uplo, trans, alpha, A, B, beta, similar(A, (n, n)))
end
function syrkx(uplo::Char, trans::Char, A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}) where T
    syrkx(uplo, trans, one(T), A, zero(T), B)
end

## hemm
for (fname, fname_64, elty) in ((:cublasZhemm_v2, :cublasZhemm_v2_64, :ComplexF64),
                                (:cublasChemm_v2, :cublasChemm_v2_64, :ComplexF32))
    @eval begin
        function hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       B::StridedCuMatrix{$elty},
                       beta::Number,
                       C::StridedCuMatrix{$elty})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
    end
end
function hemm(uplo::Char, trans::Char, alpha::Number,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    m,n = size(B)
    hemm!( uplo, trans, alpha, A, B, zero(T), similar(B, (m,n) ) )
end
function hemm(uplo::Char, trans::Char, A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    hemm(uplo, trans, one(T), A, B)
end

## herk
for (fname, fname_64, elty) in ((:cublasZherk_v2, :cublasZherk_v2_64, :ComplexF64),
                                (:cublasCherk_v2, :cublasCherk_v2_64, :ComplexF32))
    @eval begin
        function herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::StridedCuVecOrMat{$elty},
                       beta::Real,
                       C::StridedCuMatrix{$elty})
            mC, n = size(C)
            if mC != n throw(DimensionMismatch("C must be square")) end
            nn = size(A, trans == 'N' ? 1 : 2)
            if nn != n throw(DimensionMismatch("herk!")) end
            k  = size(A, trans == 'N' ? 2 : 1)
            lda = max(1,stride(A,2))
            ldc = max(1,stride(C,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            else
                $fname(handle(), uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
            end
            C
        end
   end
end
function herk(uplo::Char, trans::Char, alpha::Real, A::StridedCuVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    herk!(uplo, trans, alpha, A, zero(real(T)), similar(A, (n,n)))
end
function herk(uplo::Char, trans::Char, A::StridedCuVecOrMat{T}) where T
    herk(uplo, trans, one(real(T)), A)
end

## syr2k
for (fname, fname_64, elty) in ((:cublasDsyr2k_v2, :cublasDsyr2k_v2_64, :Float64),
                                (:cublasSsyr2k_v2, :cublasSsyr2k_v2_64, :Float32),
                                (:cublasZsyr2k_v2, :cublasZsyr2k_v2_64, :ComplexF64),
                                (:cublasCsyr2k_v2, :cublasCsyr2k_v2_64, :ComplexF32))
    @eval begin
        function syr2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::StridedCuVecOrMat{$elty},
                        B::StridedCuVecOrMat{$elty},
                        beta::Number,
                        C::StridedCuMatrix{$elty})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
    end
end
function syr2k(uplo::Char,
               trans::Char,
               alpha::Number,
               A::StridedCuVecOrMat,
               B::StridedCuVecOrMat)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    syr2k!(uplo, trans, convert(T,alpha), A, B, zero(T), similar(A, T, (n, n)))
end
function syr2k(uplo::Char, trans::Char, A::StridedCuVecOrMat, B::StridedCuVecOrMat)
    syr2k(uplo, trans, one(eltype(A)), A, B)
end

## her2k
for (fname, fname_64, elty) in ((:cublasZher2k_v2, :cublasZher2k_v2_64, :ComplexF64),
                                (:cublasCher2k_v2, :cublasCher2k_v2_64, :ComplexF32))
    @eval begin
        function her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::StridedCuVecOrMat{$elty},
                        B::StridedCuVecOrMat{$elty},
                        beta::Real,
                        C::StridedCuMatrix{$elty})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            else
                $fname(handle(), uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            end
            C
        end
   end
end
function her2k(uplo::Char, trans::Char, alpha::Number,
               A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    her2k!(uplo, trans, alpha, A, B, zero(real(T)), similar(A, (n,n)))
end
function her2k(uplo::Char, trans::Char,
               A::StridedCuVecOrMat{T}, B::StridedCuVecOrMat{T}) where T
    her2k(uplo, trans, one(T), A, B)
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
                       A::StridedCuMatrix{$elty},
                       B::StridedCuMatrix{$elty},
                       C::StridedCuMatrix{$elty})
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

        function trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       B::StridedCuMatrix{$elty})
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
    end
end
function trmm(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
end
function trsm(side::Char, uplo::Char, transa::Char, diag::Char,alpha::Number,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    trsm!(side, uplo, transa, diag, alpha, A, copy(B))
end

## (TR) triangular triangular matrix solution batched
for (fname, fname_64, elty) in ((:cublasDtrsmBatched, :cublasDtrsmBatched_64, :Float64),
                                (:cublasStrsmBatched, :cublasStrsmBatched_64, :Float32),
                                (:cublasZtrsmBatched, :cublasZtrsmBatched_64, :ComplexF64),
                                (:cublasCtrsmBatched, :cublasCtrsmBatched_64, :ComplexF32))
    @eval begin
        function trsm_batched!(side::Char,
                               uplo::Char,
                               transa::Char,
                               diag::Char,
                               alpha::Number,
                               A::Vector{<:StridedCuMatrix{$elty}},
                               B::Vector{<:StridedCuMatrix{$elty}})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), side, uplo, transa, diag, m, n, alpha, Aptrs, lda, Bptrs, ldb, length(A))
            else
                $fname(handle(), side, uplo, transa, diag, m, n, alpha, Aptrs, lda, Bptrs, ldb, length(A))
            end
            unsafe_free!(Bptrs)
            unsafe_free!(Aptrs)

            B
        end
    end
end
function trsm_batched(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
                      A::Vector{<:StridedCuMatrix{T}}, B::Vector{<:StridedCuMatrix{T}}) where T
    trsm_batched!(side, uplo, transa, diag, alpha, A, copy(B) )
end

# TODO: julia, tr{m,s}m, Char -> Char
# TODO: julia, trmm!, alpha::Number -> alpha::$elty

# BLAS-like extensions
## geam
for (fname, fname_64, elty) in ((:cublasDgeam, :cublasDgeam_64, :Float64),
                                (:cublasSgeam, :cublasSgeam_64, :Float32),
                                (:cublasZgeam, :cublasZgeam_64, :ComplexF64),
                                (:cublasCgeam, :cublasCgeam_64, :ComplexF32))
    @eval begin
        function geam!(transa::Char,
                       transb::Char,
                       alpha::Number,
                       A::StridedCuMatrix{$elty},
                       beta::Number,
                       B::StridedCuMatrix{$elty},
                       C::StridedCuMatrix{$elty})
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
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
            else
                $fname(handle(), transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc)
            end
            C
        end
    end
end
function geam(transa::Char, transb::Char, alpha::Number, A::StridedCuMatrix{T},
              beta::Number, B::StridedCuMatrix{T}) where T
    m,n = size(B)
    if transb == 'T' || transb == 'C'
        geam!(transa, transb, alpha, A, beta, B, similar(B, (n,m) ) )
    elseif transb == 'N'
        geam!(transa, transb, alpha, A, beta, B, similar(B, (m,n) ) )
    end
end
function geam(uplo::Char, trans::Char,
              A::StridedCuMatrix{T}, B::StridedCuMatrix{T}) where T
    geam(uplo, trans, one(T), A, one(T), B)
end

## getrfBatched - performs LU factorizations

for (fname, elty) in ((:cublasDgetrfBatched, :Float64),
                      (:cublasSgetrfBatched, :Float32),
                      (:cublasZgetrfBatched, :ComplexF64),
                      (:cublasCgetrfBatched, :ComplexF32))
    @eval begin
        function getrf_batched!(n, ptrs::CuVector{CuPtr{$elty}}, lda, pivot::TP, _info::TI) where
                    {TP<:Union{Bool,DenseCuArray{<:Any, 2}},
                     TI<:Union{Nothing,CuArray{Cint}}}
            batchSize = length(ptrs)
            info = TI<:Nothing ? CuArray{Cint}(undef, batchSize) : _info
            finalizer(unsafe_free!, ptrs)
            if TP<:DenseCuArray
                $fname(handle(), n, ptrs, lda, pivot, info, batchSize)
                return pivot, info
            else
                if pivot
                    pivotArray = CuArray{Cint}(undef, (n, batchSize))
                    $fname(handle(), n, ptrs, lda, pivotArray, info, batchSize)
                else
                    $fname(handle(), n, ptrs, lda, CU_NULL, info, batchSize)
                    pivotArray = CUDA.zeros(Cint, (n, batchSize))
                end
                return pivotArray, info
            end
        end
    end
end

function getrf_batched!(A::Vector{<:StridedCuMatrix},
                        pivot::Union{Bool,DenseCuArray{<:Any, 2}},
                        info::Union{Nothing,CuArray{Cint}}=nothing)
    for As in A
        m,n = size(As)
        if m != n
            throw(DimensionMismatch("All matrices must be square!"))
        end
    end
    m,n = size(A[1])
    lda = max(1,stride(A[1],2))

    Aptrs = unsafe_batch(A)
    return getrf_batched!(n, Aptrs, lda, pivot, info)..., A
end
function getrf_batched(A::Vector{<:StridedCuMatrix},
                       pivot::Union{Bool,DenseCuArray{<:Any, 2}},
                       info::Union{Nothing,CuArray{Cint}}=nothing)
    getrf_batched!(deepcopy(A), pivot, info)
end

# CUDA has no strided batched getrf, but we can at least avoid constructing costly views
function getrf_strided_batched!(A::DenseCuArray{<:Any, 3},
                                pivot::Union{Bool,DenseCuArray{<:Any, 2}},
                                info::Union{Nothing,CuArray{Cint}}=nothing)
    m,n = size(A,1), size(A,2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    lda = max(1,stride(A,2))

    Aptrs = unsafe_strided_batch(A)
    return getrf_batched!(n, Aptrs, lda, pivot, info)..., A
end
function getrf_strided_batched(A::DenseCuArray{<:Any, 3},
                               pivot::Union{Bool,DenseCuArray{<:Any, 2}},
                               info::Union{Nothing,CuArray{Cint}}=nothing)
    getrf_strided_batched!(copy(A), pivot, info)
end


## getrsBatched - solves system of linear equations

for (fname, elty) in ((:cublasDgetrsBatched, :Float64),
                      (:cublasSgetrsBatched, :Float32),
                      (:cublasZgetrsBatched, :ComplexF64),
                      (:cublasCgetrsBatched, :ComplexF32))
    @eval begin
        function getrs_batched!(trans::Char,
                                n, nrhs,
                                Aptrs::CuVector{CuPtr{$elty}}, lda,
                                pivotArray::CuPtr,
                                Bptrs::CuVector{CuPtr{$elty}}, ldb)
            batchSize = length(Aptrs)
            info = Ref{Cint}()
            $fname(handle(), trans, n, nrhs, Aptrs, lda, pivotArray, Bptrs, ldb, info, batchSize)
            unsafe_free!(Aptrs)
            unsafe_free!(Bptrs)

            return info
        end
    end
end

function getrs_batched!(trans::Char,
                        A::Vector{<:StridedCuMatrix},
                        B::Vector{<:StridedCuMatrix},
                        pivotArray::T=nothing) where T<:Union{Nothing,CuMatrix{Cint}}
    for (As,Bs) in zip(A,B)
        m,n = size(As)
        if m != n
            throw(DimensionMismatch("All A matrices must be square!"))
        end
        o = size(Bs,1)
        if m != o
            throw(DimensionMismatch("Rows in A and B must be equal!"))
        end
    end
    m,n = size(A[1])
    lda = max(1,stride(A[1],2))
    ldb = max(1,stride(B[1],2))
    nrhs = size(B[1],2)

    Aptrs = unsafe_batch(A)
    Bptrs = unsafe_batch(B)
    pivotptr = T==Nothing ? CU_NULL : pointer(pivotArray)
    return getrs_batched!(trans, n, nrhs, Aptrs, lda, pivotptr, Bptrs, ldb), B
end
function getrs_batched(trans::Char,
                       A::Vector{<:StridedCuMatrix},
                       B::Vector{<:StridedCuMatrix},
                       pivotArray::Union{Nothing,CuMatrix{Cint}}=nothing)
    getrs_batched!(trans, A, deepcopy(B), pivotArray)
end

# CUDA has no strided batched getrs, but we can at least avoid constructing costly views
function getrs_strided_batched!(trans::Char,
                                A::DenseCuArray{<:Any, 3},
                                B::DenseCuArray{<:Any, 3},
                                pivotArray::T=nothing) where T<:Union{Nothing,CuMatrix{Cint}}
    m,n = size(A,1), size(A,2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    o = size(B,1)
    if m != o
        throw(DimensionMismatch("Rows in A and B must be equal!"))
    end
    lda = max(1,stride(A,2))
    ldb = max(1,stride(B,2))
    nrhs = size(B,2)

    Aptrs = unsafe_strided_batch(A)
    Bptrs = unsafe_strided_batch(B)
    pivotptr = T==Nothing ? CU_NULL : pointer(pivotArray)
    return getrs_batched!(trans, n, nrhs, Aptrs, lda, pivotptr, Bptrs, ldb), B
end
function getrs_strided_batched(trans::Char,
                               A::DenseCuArray{<:Any, 3},
                               B::DenseCuArray{<:Any, 3},
                               pivotArray::Union{Nothing,CuMatrix{Cint}}=nothing)
    getrs_strided_batched!(trans, A, copy(B), pivotArray)
end


## getriBatched - performs batched matrix inversion
for (fname, elty) in ((:cublasDgetriBatched, :Float64),
                      (:cublasSgetriBatched, :Float32),
                      (:cublasZgetriBatched, :ComplexF64),
                      (:cublasCgetriBatched, :ComplexF32))
    @eval begin
        function getri_batched(A::Vector{<:StridedCuMatrix{$elty}},
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

        function getri_batched!(n, Aptrs::CuVector{CuPtr{$elty}},
                          lda, Cptrs::CuVector{CuPtr{$elty}},ldc,
                          pivotArray::CuArray{Cint})
            batchSize = length(Aptrs)
            info = CuArray{Cint}(undef, batchSize)
            $fname(handle(), n, Aptrs, lda, pivotArray, Cptrs, ldc, info, batchSize)
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)
            return info
        end

        function getri_batched!(A::Vector{<:StridedCuMatrix{$elty}},
                                C::Vector{<:StridedCuMatrix{$elty}},
                                pivotArray::CuMatrix{Cint})
            n = size(A[1])[1]
            lda = max(1, stride(A[1], 2))
            ldc = max(1, stride(C[1], 2))
            Aptrs = unsafe_batch(A)
            Cptrs = unsafe_batch(C)
            info = CuArrays.zeros(Cint, length(A))
            $fname(handle(), n, Aptrs, lda, pivotArray, Cptrs, ldc, info, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)

            return info
        end
    end
end

# CUDA has no strided batched getri, but we can at least avoid constructing costly views (based on getrf_strided_batch)
function getri_strided_batched!(A::CuArray{<:Any,3}, C::CuArray{<:Any,3}, pivot::CuArray{Cint})
    m, n = size(A, 1), size(A, 2)
    if m != n
        throw(DimensionMismatch("All matrices must be square!"))
    end
    ldc = max(1, stride(C, 2))
    lda = max(1, stride(A, 2))
    Cptrs = unsafe_strided_batch(C)
    Aptrs = unsafe_strided_batch(A)
    return getri_batched!(n, Aptrs, lda, Cptrs, ldc, pivot)
end

## matinvBatched - performs batched matrix inversion

for (fname, elty) in
        ((:cublasDmatinvBatched,:Float64),
         (:cublasSmatinvBatched,:Float32),
         (:cublasZmatinvBatched,:ComplexF64),
         (:cublasCmatinvBatched,:ComplexF32))
    @eval begin
        function matinv_batched(A::Vector{<:StridedCuMatrix{$elty}})
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
for (fname, elty) in ((:cublasDgeqrfBatched, :Float64),
                      (:cublasSgeqrfBatched, :Float32),
                      (:cublasZgeqrfBatched, :ComplexF64),
                      (:cublasCgeqrfBatched, :ComplexF32))
    @eval begin
        function geqrf_batched!(A::Vector{<:StridedCuMatrix{$elty}})
            m,n = size(A[1])
            lda = max(1,stride(A[1],2))
            Aptrs = unsafe_batch(A)
            hTauArray = [zeros($elty, min(m,n)) for i in 1:length(A)]
            TauArray = CuArray{$elty,1}[]
            for i in 1:length(A)
                push!(TauArray, CuArray(hTauArray[i]))
            end
            Tauptrs = unsafe_batch(TauArray)
            info = Ref{Cint}()
            $fname(handle(), m, n, Aptrs, lda, Tauptrs, info, length(A))
            unsafe_free!(Tauptrs)

            if info[] != 0
                throw(ArgumentError,string("Invalid value at ",-info[]))
            end

            TauArray, A
        end
    end
end
function geqrf_batched(A::Vector{<:CuMatrix})
    geqrf_batched!(copy(A))
end

## gelsBatched - performs batched least squares
for (fname, elty) in ((:cublasDgelsBatched, :Float64),
                      (:cublasSgelsBatched, :Float32),
                      (:cublasZgelsBatched, :ComplexF64),
                      (:cublasCgelsBatched, :ComplexF32))
    @eval begin
        function gels_batched!(trans::Char,
                              A::Vector{<:StridedCuMatrix{$elty}},
                              C::Vector{<:StridedCuMatrix{$elty}})
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
            info  = Ref{Cint}()
            infoarray = CUDA.zeros(Cint, length(A))
            $fname(handle(), trans, m, n, nrhs, Aptrs, lda, Cptrs, ldc, info, infoarray, length(A))
            unsafe_free!(Cptrs)
            unsafe_free!(Aptrs)

            if info[] != 0
                throw(ArgumentError,string("Invalid value at ",-info[]))
            end

            A, C, infoarray
        end
    end
end
function gels_batched(trans::Char, A::Vector{<:CuMatrix{T}}, C::Vector{<:CuMatrix{T}}) where T
    gels_batched!(trans, deepcopy(A), deepcopy(C))
end

## dgmm
for (fname, fname_64, elty) in ((:cublasDdgmm, :cublasDdgmm_64, :Float64),
                                (:cublasSdgmm, :cublasSdgmm_64, :Float32),
                                (:cublasZdgmm, :cublasZdgmm_64, :ComplexF64),
                                (:cublasCdgmm, :cublasCdgmm_64, :ComplexF32))
    @eval begin
        function dgmm!(mode::Char,
                       A::StridedCuMatrix{$elty},
                       X::StridedCuVector{$elty},
                       C::StridedCuMatrix{$elty})
            m, n = size(C)
            mA, nA = size(A)
            lx = length(X)
            if ((mA != m) || (nA != n )) throw(DimensionMismatch("")) end
            if ((mode == 'L') && (lx != m)) throw(DimensionMismatch("")) end
            if ((mode == 'R') && (lx != n)) throw(DimensionMismatch("")) end
            lda = max(1,stride(A,2))
            incx = stride(X,1)
            ldc = max(1,stride(C,2))
            if CUBLAS.version() >= v"12.0"
                $fname_64(handle(), mode, m, n, A, lda, X, incx, C, ldc)
            else
                $fname(handle(), mode, m, n, A, lda, X, incx, C, ldc)
            end
            C
        end
    end
end
function dgmm(mode::Char, A::StridedCuMatrix{T}, X::StridedCuVector{T}) where T
    m,n = size(A)
    dgmm!( mode, A, X, similar(A, (m,n) ) )
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
                       A::Union{StridedCuVecOrMat{$elty}, StridedVecOrMat{$elty}},
                       B::Union{StridedCuVecOrMat{$elty}, StridedVecOrMat{$elty}},
                       beta::Number,
                       C::Union{StridedCuVecOrMat{$elty}, StridedVecOrMat{$elty}})
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
    end
end
function xt_gemm(transA::Char,
                 transB::Char,
                 alpha::Number,
                 A::Union{StridedCuVecOrMat{T}, StridedVecOrMat{T}},
                 B::Union{StridedCuVecOrMat{T}, StridedVecOrMat{T}}) where T
    xt_gemm!(transA, transB, alpha, A, B, zero(T),
            similar(B, (size(A, transA == 'N' ? 1 : 2),
                        size(B, transB == 'N' ? 2 : 1))))
end
function xt_gemm(transA::Char,
                 transB::Char,
                 A::Union{StridedCuVecOrMat{T}, StridedVecOrMat{T}},
                 B::Union{StridedCuVecOrMat{T}, StridedVecOrMat{T}}) where T
    xt_gemm(transA, transB, one(T), A, B)
end

for (fname, elty) in ((:cublasXtZhemm,:ComplexF64),
                      (:cublasXtChemm,:ComplexF32))
    @eval begin
        function xt_hemm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       B::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       beta::Number,
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
    end
end
function xt_hemm(uplo::Char,
                 trans::Char,
                 alpha::Number,
                 A::Union{StridedMatrix{T}, StridedCuMatrix{T}},
                 B::Union{StridedMatrix{T}, StridedCuMatrix{T}}) where T
    m,n = size(B)
    xt_hemm!(uplo, trans, alpha, A, B, zero(T), similar(B, (m,n) ))
end
xt_hemm(uplo::Char, trans::Char,
        A::Union{StridedMatrix{T}, StridedCuMatrix{T}},
        B::Union{StridedMatrix{T}, StridedCuMatrix{T}}) where T =
    xt_hemm(uplo, trans, one(T), A, B)

for (fname, elty) in ((:cublasXtDsymm,:Float64),
                      (:cublasXtSsymm,:Float32),
                      (:cublasXtZsymm,:ComplexF64),
                      (:cublasXtCsymm,:ComplexF32))
    # TODO: fix julia dimension checks in symm!
    @eval begin
        function xt_symm!(side::Char,
                       uplo::Char,
                       alpha::Number,
                       A::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       B::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       beta::Number,
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
    end
end
function xt_symm(side::Char, uplo::Char, alpha::Number,
                 A::Union{StridedMatrix{T}, StridedCuMatrix{T}},
                 B::Union{StridedMatrix{T}, StridedCuMatrix{T}}) where T
    xt_symm!(side, uplo, alpha, A, B, zero(T), similar(B))
end
function xt_symm(side::Char, uplo::Char,
                 A::Union{StridedMatrix{T}, StridedCuMatrix{T}},
                 B::Union{StridedMatrix{T}, StridedCuMatrix{T}}) where T
    xt_symm(side, uplo, one(T), A, B)
end

for (fname, elty) in ((:cublasXtDsyrk,:Float64),
                      (:cublasXtSsyrk,:Float32),
                      (:cublasXtZsyrk,:ComplexF64),
                      (:cublasXtCsyrk,:ComplexF32))
    @eval begin
        function xt_syrk!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                       beta::Number,
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
function xt_syrk(uplo::Char, trans::Char, alpha::Number,
                 A::Union{StridedVecOrMat, StridedCuVecOrMat})
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    xt_syrk!(uplo, trans, alpha, A, zero(T), similar(A, T, (n, n)))
end
xt_syrk(uplo::Char, trans::Char, A::Union{StridedVecOrMat, StridedCuVecOrMat}) =
    xt_syrk(uplo, trans, one(eltype(A)), A)

for (fname, elty) in ((:cublasXtDsyrkx,:Float64),
                      (:cublasXtSsyrkx,:Float32),
                      (:cublasXtZsyrkx,:ComplexF64),
                      (:cublasXtCsyrkx,:ComplexF32))
    @eval begin
        function xt_syrkx!(uplo::Char,
                       trans::Char,
                       alpha::Number,
                       A::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                       B::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                       beta::Number,
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
function xt_syrkx(uplo::Char, trans::Char, alpha::Number,
              A::Union{StridedVecOrMat, StridedCuVecOrMat},
              B::Union{StridedVecOrMat, StridedCuVecOrMat},
              beta::Number)
    T = eltype(A)
    n = size(A, trans == 'N' ? 1 : 2)
    xt_syrkx!(uplo, trans, alpha, A, B, beta, similar(A, T, (n, n)))
end
xt_syrkx(uplo::Char, trans::Char,
         A::Union{StridedVecOrMat, StridedCuVecOrMat},
         B::Union{StridedVecOrMat, StridedCuVecOrMat}) =
    xt_syrkx(uplo, trans, one(eltype(A)), A, B, zero(eltype(B)))

for (fname, elty) in ((:cublasXtZherk,:ComplexF64),
                      (:cublasXtCherk,:ComplexF32))
    @eval begin
        function xt_herk!(uplo::Char,
                       trans::Char,
                       alpha::Real,
                       A::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                       beta::Real,
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
   end
end
function xt_herk(uplo::Char, trans::Char, alpha::Real,
                 A::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    xt_herk!(uplo, trans, alpha, A, real(zero(T)), similar(A, T, (n,n)))
end
function xt_herk(uplo::Char, trans::Char,
                 A::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}}) where T
    xt_herk(uplo, trans, real(one(T)), A)
end

for (fname, elty) in ((:cublasXtZher2k,:ComplexF64),
                      (:cublasXtCher2k,:ComplexF32))
    @eval begin
        function xt_her2k!(uplo::Char,
                        trans::Char,
                        alpha::Number,
                        A::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                        B::Union{StridedVecOrMat{$elty}, StridedCuVecOrMat{$elty}},
                        beta::Real,
                        C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
    end
end

function xt_her2k(uplo::Char, trans::Char, alpha::Number,
                  A::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}},
                  B::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}}) where T
    n = size(A, trans == 'N' ? 1 : 2)
    xt_her2k!(uplo, trans, alpha, A, B, zero(real(T)), similar(A, (n,n)))
end
function xt_her2k(uplo::Char, trans::Char,
                  A::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}},
                  B::Union{StridedVecOrMat{T}, StridedCuVecOrMat{T}}) where T
    xt_her2k(uplo, trans, one(T), A, B)
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
                       A::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       B::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}},
                       C::Union{StridedMatrix{$elty}, StridedCuMatrix{$elty}})
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
        function xt_trsm!(side::Char,
                       uplo::Char,
                       transa::Char,
                       diag::Char,
                       alpha::Number,
                       A::Union{StridedCuMatrix{$elty}, StridedMatrix{$elty}},
                       B::Union{StridedCuMatrix{$elty}, StridedMatrix{$elty}})
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
    end
end
function xt_trmm(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
                 A::Union{StridedCuMatrix{T}, StridedMatrix{T}},
                 B::Union{StridedCuMatrix{T}, StridedMatrix{T}}) where T
    xt_trmm!(side, uplo, transa, diag, alpha, A, B, similar(B))
end
function xt_trsm(side::Char, uplo::Char, transa::Char, diag::Char, alpha::Number,
                 A::Union{StridedCuMatrix{T}, StridedMatrix{T}},
                 B::Union{StridedCuMatrix{T}, StridedMatrix{T}}) where T
    # TODO: better way to perform synchronous copy
    xt_trsm!(side, uplo, transa, diag, alpha, A, @sync(copy(B)))
end

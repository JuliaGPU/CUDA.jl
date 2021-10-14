export symrcm, symmdq, symamd, metisnd, zfd

using SparseArrays

using ..CUSPARSE: CuSparseMatrixCSR, CuSparseMatrixCSC, CuMatrixDescriptor

function cusolverSpCreate()
  handle_ref = Ref{cusolverSpHandle_t}()
  @check unsafe_cusolverSpCreate(handle_ref) CUSOLVER_STATUS_NOT_INITIALIZED
  return handle_ref[]
end

function cusolverMgCreate()
  handle_ref = Ref{cusolverMgHandle_t}()
  res = @retry_reclaim err->isequal(err, CUSOLVER_STATUS_ALLOC_FAILED) ||
                            isequal(err, CUSOLVER_STATUS_NOT_INITIALIZED) begin
    unsafe_cusolverMgCreate(handle_ref)
  end
  if res != CUSOLVER_STATUS_SUCCESS
    throw_api_error(res)
  end
  return handle_ref[]
end

#csrlsvlu
for (fname, elty, relty) in ((:cusolverSpScsrlsvluHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsvluHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsvluHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvluHost, :ComplexF64, :Float64))
    @eval begin
        function csrlsvlu!(A::SparseMatrixCSC{$elty},
                           b::Vector{$elty},
                           x::Vector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("LU factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            Mat     = similar(A)
            transpose!(Mat, A)
            desca = CuMatrixDescriptor('G', 'L', 'N', inda)
            singularity = Ref{Cint}(1)
            $fname(sparse_handle(), n, length(A.nzval), desca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval), b,
                   tol, reorder, x, singularity)

            if singularity[] != -1
                throw(SingularException(singularity[]))
            end

            x
        end
    end
end

#csrlsvqr
for (fname, elty, relty) in ((:cusolverSpScsrlsvqr, :Float32, :Float32),
                             (:cusolverSpDcsrlsvqr, :Float64, :Float64),
                             (:cusolverSpCcsrlsvqr, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvqr, :ComplexF64, :Float64))
    @eval begin
        function csrlsvqr!(A::CuSparseMatrixCSR{$elty},
                           b::CuVector{$elty},
                           x::CuVector{$elty},
                           tol::$relty,
                           reorder::Cint,
                           inda::Char)
            n = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("QR factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            desca = CuMatrixDescriptor('G', 'L', 'N', inda)
            singularity = Ref{Cint}(1)
            $fname(sparse_handle(), n, A.nnz, desca, A.nzVal, A.rowPtr, A.colVal, b, tol, reorder, x, singularity)

            if singularity[] != -1
                throw(SingularException(singularity[]))
            end

            x
        end
    end
end

#csrlsvchol
for (fname, elty, relty) in ((:cusolverSpScsrlsvchol, :Float32, :Float32),
                             (:cusolverSpDcsrlsvchol, :Float64, :Float64),
                             (:cusolverSpCcsrlsvchol, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsvchol, :ComplexF64, :Float64))
    @eval begin
        function csrlsvchol!(A::CuSparseMatrixCSR{$elty},
                             b::CuVector{$elty},
                             x::CuVector{$elty},
                             tol::$relty,
                             reorder::Cint,
                             inda::Char)
            n      = size(A,1)
            if size(A,2) != n
                throw(DimensionMismatch("Cholesky factorization is only possible for square matrices!"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            desca = CuMatrixDescriptor('G', 'L', 'N', inda)
            singularity = zeros(Cint,1)
            $fname(sparse_handle(), n, A.nnz, desca, A.nzVal, A.rowPtr, A.colVal, b, tol, reorder, x, singularity)

            if singularity[1] != -1
                throw(SingularException(singularity[1]))
            end

            x
        end
    end
end

#csrlsqvqr
for (fname, elty, relty) in ((:cusolverSpScsrlsqvqrHost, :Float32, :Float32),
                             (:cusolverSpDcsrlsqvqrHost, :Float64, :Float64),
                             (:cusolverSpCcsrlsqvqrHost, :ComplexF32, :Float32),
                             (:cusolverSpZcsrlsqvqrHost, :ComplexF64, :Float64))
    @eval begin
        function csrlsqvqr!(A::SparseMatrixCSC{$elty},
                            b::Vector{$elty},
                            x::Vector{$elty},
                            tol::$relty,
                            inda::Char)
            m,n    = size(A)
            if m < n
                throw(DimensionMismatch("csrlsqvqr only works when the first dimension of A, $m, is greater than or equal to the second dimension of A, $n"))
            end
            if size(A,2) != length(b)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of b, $(length(b))"))
            end
            if length(x) != length(b)
                throw(DimensionMismatch("length of x, $(length(x)), must match the length of b, $(length(b))"))
            end

            desca  = CuMatrixDescriptor('G', 'L', 'N', inda)
            p        = zeros(Cint,n)
            min_norm = zeros($relty,1)
            rankA    = zeros(Cint,1)
            Mat      = similar(A)
            transpose!(Mat, A)
            $fname(sparse_handle(), m, n, length(A.nzval), desca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval),
                   b, tol, rankA, x, p, min_norm)

            x, rankA[1], p, min_norm[1]
        end
    end
end

#csreigvsi
for (fname, elty, relty) in ((:cusolverSpScsreigvsi, :Float32, :Float32),
                             (:cusolverSpDcsreigvsi, :Float64, :Float64),
                             (:cusolverSpCcsreigvsi, :ComplexF32, :Float32),
                             (:cusolverSpZcsreigvsi, :ComplexF64, :Float64))
    @eval begin
        function csreigvsi(A::CuSparseMatrixCSR{$elty},
                           μ_0::$elty,
                           x_0::CuVector{$elty},
                           tol::$relty,
                           maxite::Cint,
                           inda::Char)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end
            if n != length(x_0)
                throw(DimensionMismatch("second dimension of A, $(size(A,2)), must match the length of x_0, $(length(x_0))"))
            end

            desca = CuMatrixDescriptor('G', 'L', 'N', inda)
            x       = copy(x_0)
            μ       = CUDA.zeros($elty,1)
            $fname(sparse_handle(), n, A.nnz, desca, A.nzVal, A.rowPtr, A.colVal,
                   μ_0, x_0, maxite, tol, μ, x)

            collect(μ)[1], x
        end
    end
end

#csreigs
for (fname, elty, relty) in ((:cusolverSpScsreigsHost, :ComplexF32, :Float32),
                             (:cusolverSpDcsreigsHost, :ComplexF64, :Float64),
                             (:cusolverSpCcsreigsHost, :ComplexF32, :ComplexF32),
                             (:cusolverSpZcsreigsHost, :ComplexF64, :ComplexF64))
    @eval begin
        function csreigs(A::SparseMatrixCSC{$relty},
                         lbc::$elty,
                         ruc::$elty,
                         inda::Char)
            m,n    = size(A)
            if m != n
                throw(DimensionMismatch("A must be square!"))
            end

            desca = CuMatrixDescriptor('G', 'L', 'N', inda)
            numeigs = Ref{Cint}(0)
            Mat     = similar(A)
            transpose!(Mat, A)
            $fname(sparse_handle(), n, length(A.nzval), desca, Mat.nzval,
                   convert(Vector{Cint},Mat.colptr), convert(Vector{Cint},Mat.rowval),
                   lbc, ruc, numeigs)

            numeigs[]
        end
    end
end

#csrsymrcm
function symrcm(A::SparseMatrixCSC, index::Char)
    n, m = size(A)
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    nnzA = nnz(A)
    Mat = similar(A)
    transpose!(Mat, A)
    colsA = convert(Vector{Cint}, Mat.rowval)
    rowsA = convert(Vector{Cint}, Mat.colptr)
    p = zeros(Cint, n)
    cusolverSpXcsrsymrcmHost(sparse_handle(), n, nnzA, descA, rowsA, colsA, p)
    return p
end

#csrsymmdq
function symmdq(A::SparseMatrixCSC, index::Char)
    n, m = size(A)
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    nnzA = nnz(A)
    Mat = similar(A)
    transpose!(Mat, A)
    colsA = convert(Vector{Cint}, Mat.rowval)
    rowsA = convert(Vector{Cint}, Mat.colptr)
    p = zeros(Cint, n)
    cusolverSpXcsrsymmdqHost(sparse_handle(), n, nnzA, descA, rowsA, colsA, p)
    return p
end

#csrsymamd
function symamd(A::SparseMatrixCSC, index::Char)
    n, m = size(A)
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    nnzA = nnz(A)
    Mat = similar(A)
    transpose!(Mat, A)
    colsA = convert(Vector{Cint}, Mat.rowval)
    rowsA = convert(Vector{Cint}, Mat.colptr)
    p = zeros(Cint, n)
    cusolverSpXcsrsymamdHost(sparse_handle(), n, nnzA, descA, rowsA, colsA, p)
    return p
end

#csrmetisnd
function metisnd(A::SparseMatrixCSC, index::Char)
    n, m = size(A)
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    nnzA = nnz(A)
    Mat = similar(A)
    transpose!(Mat, A)
    colsA = convert(Vector{Cint}, Mat.rowval)
    rowsA = convert(Vector{Cint}, Mat.colptr)
    p = zeros(Cint, n)
    cusolverSpXcsrmetisndHost(sparse_handle(), n, nnzA, descA, rowsA, colsA, C_NULL, p)
    return p
end

#csrzfd
for (fname, elty) in ((:cusolverSpScsrzfdHost, :Float32),
                      (:cusolverSpDcsrzfdHost, :Float64),
                      (:cusolverSpCcsrzfdHost, :ComplexF32),
                      (:cusolverSpZcsrzfdHost, :ComplexF64))
    @eval begin
        function zfd(A::SparseMatrixCSC{$elty}, index::Char)
            n, m = size(A)
            (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            nnzA = nnz(A)
            Mat = similar(A)
            transpose!(Mat, A)
            colsA = convert(Vector{Cint}, Mat.rowval)
            rowsA = convert(Vector{Cint}, Mat.colptr)
            valsA = Mat.nzval
            p = zeros(Cint, n)
            numnz = Ref{Cint}(0)
            $fname(sparse_handle(), n, nnzA, descA, valsA, rowsA, colsA, p, numnz)
            (numnz[] < n) && throw(SingularException(n - numnz[]))
            return p
        end
    end
end

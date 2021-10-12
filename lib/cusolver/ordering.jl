export symrcm, symmdq, symamd, metisnd, zfd

function symrcm(A::CuSparseMatrixCSR, index::Char)
    n, m = A.dims
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', index)
    rowsA = A.rowPtr |> Vector{Cint}
    colsA = A.colVal |> Vector{Cint}
    p = zeros(Cint, n)
    cusolverSpXcsrsymrcmHost(sparse_handle(), n, nnz(A), descA, rowsA, colsA, p)
    if index == 'O'
        p .+= Cint(1)
    end
    return p
end

function symmdq(A::CuSparseMatrixCSR, index::Char)
    n, m = A.dims
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', index)
    rowsA = A.rowPtr |> Vector{Cint}
    colsA = A.colVal |> Vector{Cint}
    p = zeros(Cint, n)
    cusolverSpXcsrsymmdqHost(sparse_handle(), n, nnz(A), descA, rowsA, colsA, p)
    if index == 'O'
        p .+= Cint(1)
    end
    return p
end

function symamd(A::CuSparseMatrixCSR, index::Char)
    n, m = A.dims
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', index)
    rowsA = A.rowPtr |> Vector{Cint}
    colsA = A.colVal |> Vector{Cint}
    p = zeros(Cint, n)
    cusolverSpXcsrsymamdHost(sparse_handle(), n, nnz(A), descA, rowsA, colsA, p)
    if index == 'O'
        p .+= Cint(1)
    end
    return p
end

function metisnd(A::CuSparseMatrixCSR, index::Char)
    n, m = A.dims
    (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
    descA = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', index)
    rowsA = A.rowPtr |> Vector{Cint}
    colsA = A.colVal |> Vector{Cint}
    p = zeros(Cint, n)
    cusolverSpXcsrmetisndHost(sparse_handle(), n, nnz(A), descA, rowsA, colsA, C_NULL, p)
    if index == 'O'
        p .+= Cint(1)
    end
    return p
end

for (fname, elty) in ((:cusolverSpScsrzfdHost, :Float32),
                      (:cusolverSpDcsrzfdHost, :Float64),
                      (:cusolverSpCcsrzfdHost, :ComplexF32),
                      (:cusolverSpZcsrzfdHost, :ComplexF64))
    @eval begin
        function zfd(A::CuSparseMatrixCSR{$elty}, index::Char)
            n, m = A.dims
            (m ≠ m) && throw(DimensionMismatch("A must be square, but has dimensions ($n,$m)!"))
            descA = CUSPARSE.CuMatrixDescriptor('G', 'L', 'N', index)
            rowsA = A.rowPtr |> Vector{Cint}
            colsA = A.colVal |> Vector{Cint}
            valsA = A.nzVal  |> Vector{$elty}
            p = zeros(Cint, n)
            numnz = Ref{Cint}(0)
            $fname(sparse_handle(), n, nnz(A), descA, valsA, colsA, valsA, p, numnz)
            (numnz[] < n) && error("A has structural singulary")
            if index == 'O'
                p .+= Cint(1)
            end
            return p
        end
    end
end

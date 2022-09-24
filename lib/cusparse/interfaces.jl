# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    n_A, m_A = (transa != 'N') ? reverse(size(A)) : size(A)
    n_B, m_B = (transb != 'N') ? reverse(size(B)) : size(B)
    n_C, m_C = size(C)
    m_A == n_B || throw(DimensionMismatch())
    n_A == n_C || throw(DimensionMismatch())
    m_B == m_C || throw(DimensionMismatch())
    isempty(B) && return CUDA.zeros(eltype(B), size(A, 1), 0)

    if version() <= v"10.3.1"
        # Generic mm! doesn't support transposed B on CUDA10
        return mm2!(transa, transb, alpha, A, B, beta, C, 'O')
    end
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

tag_wrappers = ((identity, identity),
                (T -> :(HermOrSym{T, <:$T}), A -> :(parent($A))))
op_wrappers = (
    (identity, T -> 'N', identity),
    (T -> :(Transpose{<:T, <:$T}), T -> 'T', A -> :(parent($A))),
    (T -> :(Adjoint{<:T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A)))
)
for (taga, untaga) in tag_wrappers, (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(taga(:(CuSparseMatrix{T})))

    @eval begin
        function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::DenseCuVector{T},
                                    alpha::Number, beta::Number) where {T <: Union{Float16, ComplexF16, BlasFloat}}
            mv_wrapper($transa(T), alpha, $(untaga(unwrapa(:A))), B, beta, C)
        end

        function LinearAlgebra.mul!(C::CuVector{Complex{T}}, A::$TypeA, B::DenseCuVector{Complex{T}},
                                    alpha::Number, beta::Number) where {T <: Union{Float16, BlasFloat}}
            mv_wrapper($transa(T), alpha, $(untaga(unwrapa(:A))), B, beta, C)
        end
    end

    for (tagb, untagb) in tag_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        TypeB = wrapb(tagb(:(DenseCuMatrix{T})))

        @eval begin
            function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                        alpha::Number, beta::Number) where {T <: Union{Float16, ComplexF16, BlasFloat}}
                mm_wrapper($transa(T), $transb(T), alpha, $(untaga(unwrapa(:A))), $(untagb(unwrapb(:B))), beta, C)
            end
        end
    end
end

Base.:(+)(A::CuSparseMatrixCSR, B::CuSparseMatrixCSR) = geam(one(eltype(A)), A, one(eltype(A)), B, 'O')
Base.:(-)(A::CuSparseMatrixCSR, B::CuSparseMatrixCSR) = geam(one(eltype(A)), A, -one(eltype(A)), B, 'O')

Base.:(+)(A::CuSparseMatrixCSR, B::Adjoint{T,<:CuSparseMatrixCSR}) where {T} = A + Transpose(conj(B.parent))
Base.:(-)(A::CuSparseMatrixCSR, B::Adjoint{T,<:CuSparseMatrixCSR}) where {T} = A - Transpose(conj(B.parent))
Base.:(+)(A::Adjoint{T,<:CuSparseMatrixCSR}, B::CuSparseMatrixCSR) where {T} = Transpose(conj(A.parent)) + B
Base.:(-)(A::Adjoint{T,<:CuSparseMatrixCSR}, B::CuSparseMatrixCSR) where {T} = Transpose(conj(A.parent)) - B
Base.:(+)(A::Adjoint{T,<:CuSparseMatrixCSR}, B::Adjoint{T,<:CuSparseMatrixCSR}) where {T} =
    Transpose(conj(A.parent)) + B
Base.:(-)(A::Adjoint{T,<:CuSparseMatrixCSR}, B::Adjoint{T,<:CuSparseMatrixCSR}) where {T} =
    Transpose(conj(A.parent)) - B

function Base.:(+)(A::CuSparseMatrixCSR, B::Transpose{T,<:CuSparseMatrixCSR}) where {T}
    cscB = CuSparseMatrixCSC(B.parent)
    transB = CuSparseMatrixCSR(cscB.colPtr, cscB.rowVal, cscB.nzVal, size(cscB))
    return geam(one(T), A, one(T), transB, 'O')
end

function Base.:(-)(A::CuSparseMatrixCSR, B::Transpose{T,<:CuSparseMatrixCSR}) where {T}
    cscB = CuSparseMatrixCSC(B.parent)
    transB = CuSparseMatrixCSR(cscB.colPtr, cscB.rowVal, cscB.nzVal, size(cscB))
    return geam(one(T), A, -one(T), transB, 'O')
end

function Base.:(+)(A::Transpose{T,<:CuSparseMatrixCSR}, B::CuSparseMatrixCSR) where {T}
    cscA = CuSparseMatrixCSC(A.parent)
    transA = CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, size(cscA))
    geam(one(T), transA, one(T), B, 'O')
end

function Base.:(-)(A::Transpose{T,<:CuSparseMatrixCSR}, B::CuSparseMatrixCSR) where {T}
    cscA = CuSparseMatrixCSC(A.parent)
    transA = CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, size(cscA))
    geam(one(T), transA, -one(T), B, 'O')
end

function Base.:(+)(A::Transpose{T,<:CuSparseMatrixCSR}, B::Transpose{T,<:CuSparseMatrixCSR}) where {T}
    C = geam(one(T), A.parent, one(T), B.parent, 'O')
    cscC = CuSparseMatrixCSC(C)
    return CuSparseMatrixCSR(cscC.colPtr, cscC.rowVal, cscC.nzVal, size(cscC))
end

function Base.:(-)(A::Transpose{T,<:CuSparseMatrixCSR}, B::Transpose{T,<:CuSparseMatrixCSR}) where {T}
    C = geam(one(T), A.parent, -one(T), B.parent, 'O')
    cscC = CuSparseMatrixCSC(C)
    return CuSparseMatrixCSR(cscC.colPtr, cscC.rowVal, cscC.nzVal, size(cscC))
end

function Base.:(+)(A::CuSparseMatrixCSR, B::CuSparseMatrix)
    csrB = CuSparseMatrixCSR(B)
    return geam(one(eltype(A)), A, one(eltype(A)), csrB, 'O')
end

function Base.:(-)(A::CuSparseMatrixCSR, B::CuSparseMatrix)
    csrB = CuSparseMatrixCSR(B)
    return geam(one(eltype(A)), A, -one(eltype(A)), csrB, 'O')
end

function Base.:(+)(A::CuSparseMatrix, B::CuSparseMatrixCSR)
    csrA = CuSparseMatrixCSR(A)
    return geam(one(eltype(A)), csrA, one(eltype(A)), B, 'O')
end

function Base.:(-)(A::CuSparseMatrix, B::CuSparseMatrixCSR)
    csrA = CuSparseMatrixCSR(A)
    return geam(one(eltype(A)), csrA, -one(eltype(A)), B, 'O')
end

function Base.reshape(A::CuSparseMatrixCOO{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    nrows, ncols = size(A)
    flat_indices = nrows * (A.colInd .- 1) + A.rowInd .- 1
    new_col, new_row = div.(flat_indices, dims[1]) .+ 1, rem.(flat_indices, dims[1]) .+ 1
    CuSparseMatrixCOO(sparse(new_row, new_col, A.nzVal, dims[1], dims[2]))
end
function LinearAlgebra.triu(A::CuSparseMatrixCOO{T,M}, k::Integer) where {T,M}
    mask = A.rowInd .+ k .<= A.colInd
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    datas = A.nzVal[mask]
    CuSparseMatrixCOO(sparse(rows, cols, datas, size(A, 1), size(A, 2)))
end
function LinearAlgebra.tril(A::CuSparseMatrixCOO{T,M}, k::Integer) where {T,M}
    mask = A.rowInd .+ k .>= A.colInd
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    datas = A.nzVal[mask]
    CuSparseMatrixCOO(sparse(rows, cols, datas, size(A, 1), size(A, 2)))
end
function LinearAlgebra.kron(A::CuSparseMatrixCOO{T,M}, B::CuSparseMatrixCOO{T,M}) where {T,M}
    out_shape = (size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
    Annz = Int64(A.nnz)
    Bnnz = Int64(B.nnz)

    if A.nnz == 0 || B.nnz == 0
        return CuSparseMatrixCOO(spzeros(T, out_shape))
    end

    row = (A.rowInd .- 1) .* size(B, 1)
    row = repeat(row, inner = Bnnz)
    col = (A.colInd .- 1) .* size(B, 2)
    col = repeat(col, inner = Bnnz)
    data = repeat(A.nzVal, inner = Bnnz)

    row, col = reshape(row, Annz, Bnnz), reshape(col, Annz, Bnnz)
    row .+= reshape(repeat(reverse(B.rowInd) .- 1, outer = Annz), Annz, Bnnz)
    col .+= reshape(repeat(reverse(B.colInd) .- 1, outer = Annz), Annz, Bnnz)
    row, col = reshape(row, Annz * Bnnz) .+ 1, reshape(col, Annz * Bnnz) .+ 1

    data = reshape(data, Annz, Bnnz) .* reshape(repeat(reverse(B.nzVal), outer = Annz), Annz, Bnnz)
    data = reshape(data, Annz * Bnnz)
    CuSparseMatrixCOO(sparse(row, col, data, out_shape[1], out_shape[2]))
end

for SparseMatrixType in [:CuSparseMatrixCSC, :CuSparseMatrixCSR, CuSparseMatrixBSR, CuSparseMatrixCOO]
    @eval begin
        if $SparseMatrixType in [CuSparseMatrixCSR, CuSparseMatrixBSR]
            function LinearAlgebra.mul!(Y::$SparseMatrixType{T}, A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T}
                mm!('N', 'N', one(T), A, B, zero(T), Y, 'O')
            end
            function LinearAlgebra.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T}
                Y = $SparseMatrixType(spzeros(T, size(A, 1), size(B, 2)))
                mul!(Y, A, B)
            end
        else
            function LinearAlgebra.mul!(Y::$SparseMatrixType{T}, A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T}
                Y2 = CuSparseMatrixCSR(Y)
                A2 = CuSparseMatrixCSR(A)
                B2 = CuSparseMatrixCSR(B)
                Y = $SparseMatrixType( mm!('N', 'N', one(T), A2, B2, zero(T), Y2, 'O') )
            end
            function LinearAlgebra.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T}
                Y = $SparseMatrixType(spzeros(T, size(A, 1), size(B, 2)))
                mul!(Y, A, B)
            end
        end

        if $SparseMatrixType in [CuSparseMatrixCSR] # Is it possible to put also CuSparseMatrixCSC?
            function LinearAlgebra.adjoint(A::$SparseMatrixType{T}) where {T}
                cscA = CuSparseMatrixCSC(conj(A))
                $SparseMatrixType( CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, size(cscA)) )
            end
            function LinearAlgebra.transpose(A::$SparseMatrixType{T}) where {T}
                cscA = CuSparseMatrixCSC(A)
                $SparseMatrixType( CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, size(cscA)) )
            end
        end
        
        if $SparseMatrixType != CuSparseMatrixCOO
            Base.reshape(A::$SparseMatrixType{T,M}, dims::NTuple{N,Int}) where {T,N,M} = $SparseMatrixType( reshape(CuSparseMatrixCOO(A), dims) )
            
            LinearAlgebra.triu(A::$SparseMatrixType{T,M}, k::Integer) where {T,M} = $SparseMatrixType( triu(CuSparseMatrixCOO(A), k) )
            LinearAlgebra.tril(A::$SparseMatrixType{T,M}, k::Integer) where {T,M} = $SparseMatrixType( tril(CuSparseMatrixCOO(A), k) )

            LinearAlgebra.kron(A::$SparseMatrixType{T,M}, B::$SparseMatrixType{T,M}) where {T,M} = $SparseMatrixType( kron(CuSparseMatrixCOO(A), CuSparseMatrixCOO(B)) )
        end
        LinearAlgebra.triu(A::$SparseMatrixType{T,M}) where {T,M} = triu(A, 0)
        LinearAlgebra.tril(A::$SparseMatrixType{T,M}) where {T,M} = tril(A, 0)
    end
end

# triangular

## direct
for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                            (:UnitLowerTriangular, 'L', 'U'),
                            (:UpperTriangular, 'U', 'N'),
                            (:UnitUpperTriangular, 'U', 'U'))
    @eval begin
        # Left division
        LinearAlgebra.ldiv!(A::$t{T,<:AbstractCuSparseMatrix},
                            B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!('N', $uploc, $isunitc, one(T), parent(A), B, 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:AbstractCuSparseMatrix},
                            B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!('N', 'N', $uploc, $isunitc, one(T), parent(A), B, 'O')
    end
end

## adjoint/transpose ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Left division with vectors
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!('T', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuVector{T}) where {T<:BlasReal} =
            sv2!('T', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuVector{T}) where {T<:BlasComplex} =
            sv2!('C', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        # Left division with matrices
        LinearAlgebra.ldiv!(A::$t{<:Any,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!('T', 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuMatrix{T}) where {T<:BlasReal} =
            sm2!('T', 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{<:Any,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuMatrix{T}) where {T<:BlasComplex} =
            sm2!('C', 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')
    end
end


## uniform scaling

# these operations materialize the identity matrix and re-use broadcast
# TODO: can we do without this, and just use the broadcast implementation
#       with a singleton argument it knows how to index?

function _sparse_identity(::Type{<:CuSparseMatrixCSR{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    rowPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[1]-len+1)))
    colVal = CuVector{Ti}(1:len)
    nzVal = CUDA.fill(I.λ, len)
    CuSparseMatrixCSR{Tv,Ti}(rowPtr, colVal, nzVal, dims)
end

function _sparse_identity(::Type{<:CuSparseMatrixCSC{<:Any,Ti}},
                          I::UniformScaling{Tv}, dims::Dims) where {Tv,Ti}
    len = min(dims[1], dims[2])
    colPtr = CuVector{Ti}(vcat(1:len, fill(len+1, dims[2]-len+1)))
    rowVal = CuVector{Ti}(1:len)
    nzVal = CUDA.fill(I.λ, len)
    CuSparseMatrixCSC{Tv,Ti}(colPtr, rowVal, nzVal, dims)
end

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}, J::UniformScaling) =
    A .+ _sparse_identity(typeof(A), J, size(A))

Base.:(-)(J::UniformScaling, A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) =
    _sparse_identity(typeof(A), J, size(A)) .- A

# TODO: let Broadcast handle this automatically (a la SparseArrays.PromoteToSparse)
for SparseMatrixType in [:CuSparseMatrixCSC, :CuSparseMatrixCSR], op in [:(+), :(-)]
    @eval begin
        function Base.$op(lhs::Diagonal{T,<:CuArray}, rhs::$SparseMatrixType{T}) where {T}
            return $op($SparseMatrixType(lhs), rhs)
        end
        function Base.$op(lhs::$SparseMatrixType{T}, rhs::Diagonal{T,<:CuArray}) where {T}
            return $op(lhs, $SparseMatrixType(rhs))
        end
    end
end

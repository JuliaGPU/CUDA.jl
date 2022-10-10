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

        function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::CuSparseVector{T},
                                    alpha::Number, beta::Number) where {T <: Union{Float16, ComplexF16, BlasFloat}}
            mv_wrapper($transa(T), alpha, $(untaga(unwrapa(:A))), CuVector{T}(B), beta, C)
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

for (taga, untaga) in tag_wrappers, (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(taga(:(DenseCuMatrix{T})))
    @eval begin
        function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::CuSparseVector{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
            gemvi!($transa(T), alpha, $(untaga(unwrapa(:A))), B, beta, C, 'O')
        end

        function SparseArrays.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: BlasFloat}
            m, n = size(A)
            length(x) == n || throw(DimensionMismatch())
            y = CuVector{T}(undef, m)
            mul!(y, A, x)
        end

        for (tagb, untagb) in tag_wrappers, (wrapb, transb, unwrapb) in op_wrappers
            TypeB = wrapb(tagb(:(CuSparseMatrix{T})))

            @eval begin
                # Use the relations AB = (BᵀAᵀ)ᵀ and AB = (BᴴAᴴ)ᴴ to support dense matrix - sparse matrix products
                function LinearAlgebra.:(*)(A::$TypeA, B::$TypeB) where {T <: Union{Float16, ComplexF16, BlasFloat}}
                    mA, nA = size(A)
                    mB, nB = size(B)
                    C = CuMatrix{T}(undef, nB, mA)
                    transa = $transa(T)
                    transb = $transb(T)
                    (transa == 'N' && transb == 'N') && (transa2 = 'T' ; transb2 = 'T')
                    (transa == 'T' && transb == 'N') && (transa2 = 'N' ; transb2 = 'T')
                    (transa == 'C' && transb == 'N') && (transa2 = 'N' ; transb2 = 'C')
                    (transa == 'N' && transb == 'T') && (transa2 = 'T' ; transb2 = 'N')
                    (transa == 'N' && transb == 'C') && (transa2 = 'C' ; transb2 = 'N')
                    (transa == 'T' && transb == 'T') && (transa2 = 'N' ; transb2 = 'N')
                    (transa == 'C' && transb == 'C') && (transa2 = 'N' ; transb2 = 'N')
                    (transa == 'T' && transb == 'C') && error("dense matrix - sparse matrix product is not supported with transa == 'T' and transb == 'C'")
                    (transa == 'C' && transb == 'T') && error("dense matrix - sparse matrix product is not supported with transa == 'C' and transb == 'T'")
                    mm_wrapper(transb2, transa2, one(T), $(untagb(unwrapb(:B))), $(untaga(unwrapa(:A))), zero(T), C)
                    (transa == 'C' || transa == 'C') && (conj!(C))
                    permutedims(C)
                end
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

        LinearAlgebra.ldiv!(A::$t{T,<:AbstractCuSparseMatrix},
                            B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('N', 'T', $uploc, $isunitc, one(T), parent(A), parent(B), 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:AbstractCuSparseMatrix},
                            B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('N', 'C', $uploc, $isunitc, one(T), parent(A), parent(B), 'O')
    end
end

## adjoint/transpose ('uploc' reversed)
for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                            (:UnitLowerTriangular, 'U', 'U'),
                            (:UpperTriangular, 'L', 'N'),
                            (:UnitUpperTriangular, 'L', 'U'))
    @eval begin
        # Left division with vectors
        LinearAlgebra.ldiv!(A::$t{T,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!('T', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuVector{T}) where {T<:BlasFloat} =
            sv2!('C', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        # Left division with matrices
        LinearAlgebra.ldiv!(A::$t{T,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!('T', 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('T', 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Transpose{T,<:AbstractCuSparseMatrix}},
                            B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('T', 'C', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::DenseCuMatrix{T}) where {T<:BlasFloat} =
            sm2!('C', 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('C', 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), 'O')

        LinearAlgebra.ldiv!(A::$t{T,<:Adjoint{T,<:AbstractCuSparseMatrix}},
                            B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
            sm2!('C', 'C', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), 'O')
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

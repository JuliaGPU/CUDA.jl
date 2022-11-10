# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal
export _spadjoint, _sptranspose

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

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasReal} = vv!('N', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasReal} = dot(y, x)

LinearAlgebra.dot(x::CuSparseVector{T}, y::DenseCuVector{T}) where {T <: BlasComplex} = vv!('C', x, y, 'O')
LinearAlgebra.dot(x::DenseCuVector{T}, y::CuSparseVector{T}) where {T <: BlasComplex} = conj(dot(y,x))

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

        function LinearAlgebra.:(*)(A::$TypeA, x::CuSparseVector{T}) where {T <: Union{Float16, ComplexF16, BlasFloat}}
            m, n = size(A)
            length(x) == n || throw(DimensionMismatch())
            y = CuVector{T}(undef, m)
            mul!(y, A, x)
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
    end

    for (tagb, untagb) in tag_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
            TypeB = wrapb(tagb(SparseMatrixType))

            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                        alpha::Number, beta::Number) where {T <: Union{Float16, ComplexF16, BlasFloat}}
                    mm!($transa(T), $transb(T), alpha, $(untaga(unwrapa(:A))), $(untagb(unwrapb(:B))), beta, C, 'O')
                end

                function LinearAlgebra.:(*)(A::$TypeA, B::$TypeB) where {T <: Union{Float16, ComplexF16, BlasFloat}}
                    m, n = size(A)
                    k, p = size(B)
                    n == k || throw(DimensionMismatch())
                    C = CuMatrix{T}(undef, m, p)
                    mul!(C, A, B)
                end
            end
        end
    end
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
    @eval begin
        function Base.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T <: BlasFloat}
            CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
            gemm('N', 'N', one(T), A, B, 'O')
        end
    end
end

for op in (:(+), :(-))
    @eval begin
        Base.$op(A::CuSparseVector{T}, B::CuSparseVector{T}) where {T <: BlasFloat} = axpby(one(T), A, $(op)(one(T)), B, 'O')

        Base.$op(A::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}, 
                B::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}) where {T} =
            CuSparseMatrixCOO($(op)(CuSparseMatrixCSR(A), CuSparseMatrixCSR(B)))
    end

    for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR)
        @eval begin
            Base.$op(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T <: BlasFloat} = geam(one(T), A, $(op)(one(T)), B, 'O')

            Base.$op(A::$SparseMatrixType{T}, B::Adjoint{T,<:$SparseMatrixType}) where {T <: BlasFloat} = geam(one(T), A, $(op)(one(T)), _spadjoint(parent(B)), 'O')
            Base.$op(A::Adjoint{T,<:$SparseMatrixType}, B::$SparseMatrixType{T}) where {T <: BlasFloat} = geam(one(T), _spadjoint(parent(A)), $(op)(one(T)), B, 'O')
            Base.$op(A::Adjoint{T,<:$SparseMatrixType}, B::Adjoint{T,<:$SparseMatrixType}) where {T <: BlasFloat} = geam(one(T), _spadjoint(parent(A)), $(op)(one(T)), _spadjoint(parent(B)), 'O')

            Base.$op(A::$SparseMatrixType{T}, B::Transpose{T,<:$SparseMatrixType}) where {T <: BlasFloat} = geam(one(T), A, $(op)(one(T)), _sptranspose(parent(B)), 'O')
            Base.$op(A::Transpose{T,<:$SparseMatrixType}, B::$SparseMatrixType{T}) where {T <: BlasFloat} = geam(one(T), _sptranspose(parent(A)), $(op)(one(T)), B, 'O')
            Base.$op(A::Transpose{T,<:$SparseMatrixType}, B::Transpose{T,<:$SparseMatrixType}) where {T <: BlasFloat} = geam(one(T), _sptranspose(parent(A)), $(op)(one(T)), _sptranspose(parent(B)), 'O')
        end
    end

    for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCOO, :CuSparseMatrixBSR)
        @eval begin
            function Base.$op(A::CuSparseMatrixCSR{T}, B::$SparseMatrixType{T}) where {T}
                csrB = CuSparseMatrixCSR(B)
                return geam(one(T), A, $(op)(one(T)), csrB, 'O')
            end
            function Base.$op(A::$SparseMatrixType{T}, B::CuSparseMatrixCSR{T}) where {T}
                csrA = CuSparseMatrixCSR(A)
                return geam(one(T), csrA, $(op)(one(T)), B, 'O')
            end
            function Base.$op(A::Transpose{T, CuSparseMatrixCSR{T}}, B::$SparseMatrixType{T}) where {T}
                csrB = CuSparseMatrixCSR(B)
                return geam(one(T), _sptranspose(parent(A)), $(op)(one(T)), csrB, 'O')
            end
            function Base.$op(A::$SparseMatrixType{T}, B::Transpose{T, CuSparseMatrixCSR{T}}) where {T}
                csrA = CuSparseMatrixCSR(A)
                return geam(one(T), csrA, $(op)(one(T)), _sptranspose(parent(B)), 'O')
            end
            function Base.$op(A::Adjoint{T, CuSparseMatrixCSR{T}}, B::$SparseMatrixType{T}) where {T}
                csrB = CuSparseMatrixCSR(B)
                return geam(one(T), _spadjoint(parent(A)), $(op)(one(T)), csrB, 'O')
            end
            function Base.$op(A::$SparseMatrixType{T}, B::Adjoint{T, CuSparseMatrixCSR{T}}) where {T}
                csrA = CuSparseMatrixCSR(A)
                return geam(one(T), csrA, $(op)(one(T)), _spadjoint(parent(B)), 'O')
            end

            function Base.$op(A::CuSparseMatrixCSR{T}, B::Transpose{T, $SparseMatrixType}) where {T}
                csrB = CuSparseMatrixCSR(_sptranspose(parent(B)))
                return geam(one(T), A, $(op)(one(T)), csrB, 'O')
            end
            function Base.$op(A::Transpose{T, $SparseMatrixType}, B::CuSparseMatrixCSR{T}) where {T}
                csrA = CuSparseMatrixCSR(_sptranspose(parent(A)))
                return geam(one(T), csrA, $(op)(one(T)), B, 'O')
            end

            function Base.$op(A::CuSparseMatrixCSR{T}, B::Adjoint{T, $SparseMatrixType}) where {T}
                csrB = CuSparseMatrixCSR(_spadjoint(parent(B)))
                return geam(one(T), A, $(op)(one(T)), csrB, 'O')
            end
            function Base.$op(A::Adjoint{T, $SparseMatrixType}, B::CuSparseMatrixCSR{T}) where {T}
                csrA = CuSparseMatrixCSR(_spadjoint(parent(A)))
                return geam(one(T), csrA, $(op)(one(T)), B, 'O')
            end
        end
    end
end


function Base.reshape(A::CuSparseMatrixCOO{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    nrows, ncols = size(A)
    flat_indices = nrows .* (A.colInd .- 1) .+ A.rowInd .- 1
    new_col, new_row = div.(flat_indices, dims[1]) .+ 1, rem.(flat_indices, dims[1]) .+ 1
    sparse(new_row, new_col, A.nzVal, dims[1], length(dims) == 1 ? 1 : dims[2], fmt = :coo)
end

function LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::CuSparseMatrixCSR{T,M}, 
    B::CuSparseMatrixCSR{T,M}, alpha::Number, beta::Number) where {T,M}
    CUSPARSE.version() < v"11.5.1" && throw(ErrorException("This operation is not 
                                        supported by the current CUDA version."))
    gemm!('N', 'N', alpha, A, B, beta, Y, 'O')
end
LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::CuSparseMatrixCSR{T,M}, 
    B::CuSparseMatrixCSR{T,M}) where {T,M} = mul!(Y, A, B, one(T), zero(T))

LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::Transpose{T,<:CuSparseMatrixCSR}, 
    B::CuSparseMatrixCSR{T,M}) where {T,M} = mul!(Y, _sptranspose(parent(A)), B, one(T), zero(T))
LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::Transpose{T,<:CuSparseMatrixCSR}, 
    B::Transpose{T,<:CuSparseMatrixCSR}) where {T,M} = mul!(Y, _sptranspose(parent(A)), _sptranspose(parent(B)), one(T), zero(T))
LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::CuSparseMatrixCSR{T,M}, 
    B::Transpose{T,<:CuSparseMatrixCSR}) where {T,M} = mul!(Y, A, _sptranspose(parent(B)), one(t), zero(T))

LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::Adjoint{T,<:CuSparseMatrixCSR}, 
    B::CuSparseMatrixCSR{T,M}) where {T,M} = mul!(Y, _spadjoint(parent(A)), B, one(T), zero(T))
LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::Adjoint{T,<:CuSparseMatrixCSR}, 
    B::Adjoint{T,<:CuSparseMatrixCSR}) where {T,M} = mul!(Y, _spadjoint(parent(A)), _spadjoint(parent(B)), one(T), zero(T))
LinearAlgebra.mul!(Y::CuSparseMatrixCSR{T,M}, A::CuSparseMatrixCSR{T,M}, 
    B::Adjoint{T,<:CuSparseMatrixCSR}) where {T,M} = mul!(Y, A, _spadjoint(parent(B)), one(t), zero(T))

function LinearAlgebra.mul!(Y::CuSparseMatrixCOO{T,M}, A::Union{CuSparseMatrixCOO{T,M}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}, 
    B::Union{CuSparseMatrixCOO{T,M}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}, alpha::Number, beta::Number) where {T,M}
    
    Y2 = CuSparseMatrixCSR(Y)
    A2 = CuSparseMatrixCSR(A)
    B2 = CuSparseMatrixCSR(B)
    mul!(Y2, A2, B2, alpha, beta)
    copyto!(Y, CuSparseMatrixCOO(Y2))
end
function LinearAlgebra.mul!(Y::CuSparseMatrixCSC{T,M}, A::Union{CuSparseMatrixCSC{T,M}, Transpose{T,<:CuSparseMatrixCSC}, Adjoint{T,<:CuSparseMatrixCSC}}, 
    B::Union{CuSparseMatrixCSC{T,M}, Transpose{T,<:CuSparseMatrixCSC}, Adjoint{T,<:CuSparseMatrixCSC}}, alpha::Number, beta::Number) where {T,M}
    
    Y2 = CuSparseMatrixCSR(Y)
    A2 = CuSparseMatrixCSR(A)
    B2 = CuSparseMatrixCSR(B)
    mul!(Y2, A2, B2, alpha, beta)
    copyto!(Y, CuSparseMatrixCSC(Y2))
end

LinearAlgebra.mul!(Y::CuSparseMatrixCOO{T,M}, A::Union{CuSparseMatrixCOO{T,M}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}, 
    B::Union{CuSparseMatrixCOO{T,M}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}) where {T,M} = mul!(Y, A, B, one(T), zero(T))
LinearAlgebra.mul!(Y::CuSparseMatrixCSC{T,M}, A::Union{CuSparseMatrixCSC{T,M}, Transpose{T,<:CuSparseMatrixCSC}, Adjoint{T,<:CuSparseMatrixCSC}}, 
    B::Union{CuSparseMatrixCSC{T,M}, Transpose{T,<:CuSparseMatrixCSC}, Adjoint{T,<:CuSparseMatrixCSC}}) where {T,M} = mul!(Y, A, B, one(T), zero(T))

function LinearAlgebra.:(*)(A::CuSparseMatrixCSR{T,M}, B::CuSparseMatrixCSR{T,M}) where {T,M}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not 
                                        supported by the current CUDA version."))
    gemm('N', 'N', one(T), A, B, 'O')
end
function LinearAlgebra.:(*)(A::CuSparseMatrixCSC{T,M}, B::CuSparseMatrixCSC{T,M}) where {T,M}
    A2 = CuSparseMatrixCSR(A)
    B2 = CuSparseMatrixCSR(B)
    CuSparseMatrixCSC(gemm('N', 'N', one(T), A2, B2, 'O'))
end
function LinearAlgebra.:(*)(A::CuSparseMatrixCOO{T,M}, B::CuSparseMatrixCOO{T,M}) where {T,M}
    A2 = CuSparseMatrixCSR(A)
    B2 = CuSparseMatrixCSR(B)
    CuSparseMatrixCOO(gemm('N', 'N', one(T), A2, B2, 'O'))
end

function SparseArrays.droptol!(A::CuSparseMatrixCOO{T,M}, tol::Real) where {T,M}
    mask = abs.(A.nzVal) .> tol
    rows = A.rowInd[mask]
    cols = A.colInd[mask]
    datas = A.nzVal[mask]
    B = sparse(rows, cols, datas, size(A)..., fmt = :coo)
    copyto!(A, B)
end

for SparseMatrixType in [:CuSparseMatrixCSC, :CuSparseMatrixCSR, :CuSparseMatrixCOO]
    @eval begin
        if $SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR]

            Base.reshape(A::$SparseMatrixType{T,M}, dims::NTuple{N,Int}) where {T,N,M} = 
            $SparseMatrixType( reshape(CuSparseMatrixCOO(A), dims) )

            function SparseArrays.droptol!(A::$SparseMatrixType{T,M}, tol::Real) where {T,M}
                B = copy(CuSparseMatrixCOO(A))
                droptol!(B, tol)
                copyto!(A, $SparseMatrixType(B))
            end
            
        end

        LinearAlgebra.:(*)(A::Transpose{T,<:$SparseMatrixType}, B::$SparseMatrixType{T,M}) where {T,M} = _sptranspose(parent(A)) * B
        LinearAlgebra.:(*)(A::Transpose{T,<:$SparseMatrixType}, B::Transpose{T,<:$SparseMatrixType}) where {T} = _sptranspose(parent(A)) * _sptranspose(parent(B))
        LinearAlgebra.:(*)(A::$SparseMatrixType{T,M}, B::Transpose{T,<:$SparseMatrixType}) where {T,M} = A * _sptranspose(parent(B))
        LinearAlgebra.:(*)(A::Adjoint{T,<:$SparseMatrixType}, B::$SparseMatrixType{T,M}) where {T,M} = _spadjoint(parent(A)) * B
        LinearAlgebra.:(*)(A::Adjoint{T,<:$SparseMatrixType}, B::Adjoint{T,<:$SparseMatrixType}) where {T} = _spadjoint(parent(A)) * _spadjoint(parent(B))
        LinearAlgebra.:(*)(A::$SparseMatrixType{T,M}, B::Adjoint{T,<:$SparseMatrixType}) where {T,M} = A * _spadjoint(parent(B))
    end
end

function _spadjoint(A::CuSparseMatrixCSR{T,M}) where {T,M}
    cscA = CuSparseMatrixCSC(conj(A))
    CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, reverse(size(cscA)))
end
function _sptranspose(A::CuSparseMatrixCSR{T,M}) where {T,M}
    cscA = CuSparseMatrixCSC(A)
    CuSparseMatrixCSR(cscA.colPtr, cscA.rowVal, cscA.nzVal, reverse(size(cscA)))
end
function _spadjoint(A::CuSparseMatrixCSC{T,M}) where {T,M}
    CuSparseMatrixCSC(CuSparseMatrixCSR(A.colPtr, A.rowVal, conj(A.nzVal), reverse(size(A))))
end
function _sptranspose(A::CuSparseMatrixCSC{T,M}) where {T,M}
    CuSparseMatrixCSC(CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A))))
end
function _spadjoint(A::CuSparseMatrixCOO{T,M}) where {T,M}
    sparse(A.colInd, A.rowInd, conj(A.nzVal), size(A)..., fmt = :coo)
end
function _sptranspose(A::CuSparseMatrixCOO{T,M}) where {T,M}
    sparse(A.colInd, A.rowInd, A.nzVal, size(A)..., fmt = :coo)
end


# triangular
for SparseMatrixType in (:CuSparseMatrixBSR, :CuSparseMatrixCSC, :CuSparseMatrixCSR)

    ## direct
    for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                                (:UnitLowerTriangular, 'L', 'U'),
                                (:UpperTriangular, 'U', 'N'),
                                (:UnitUpperTriangular, 'U', 'U'))
        @eval begin
            # Left division with vectors
            LinearAlgebra.ldiv!(A::$t{T,<:$SparseMatrixType},
                                B::DenseCuVector{T}) where {T<:BlasFloat} =
                sv2!('N', $uploc, $isunitc, one(T), parent(A), B, 'O')

            # Left division with matrices
            LinearAlgebra.ldiv!(A::$t{T,<:$SparseMatrixType},
                                B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                sm2!('N', 'N', $uploc, $isunitc, one(T), parent(A), B, 'O')
        end
    end

    ## adjoint/transpose ('uploc' reversed)
    for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                                (:UnitLowerTriangular, 'U', 'U'),
                                (:UpperTriangular, 'L', 'N'),
                                (:UnitUpperTriangular, 'L', 'U'))

        for (opa, transa) in ((:Transpose, 'T'),
                              (:Adjoint, 'C'))
            @eval begin
                # Left division with vectors
                LinearAlgebra.ldiv!(A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuVector{T}) where {T<:BlasFloat} =
                    sv2!($transa, $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')

                # Left division with matrices
                LinearAlgebra.ldiv!(A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                    sm2!($transa, 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, 'O')
            end
        end
    end
end

for SparseMatrixType in (:CuSparseMatrixCOO,)

    ## direct
    for (t, uploc, isunitc) in ((:LowerTriangular, 'L', 'N'),
                                (:UnitLowerTriangular, 'L', 'U'),
                                (:UpperTriangular, 'U', 'N'),
                                (:UnitUpperTriangular, 'U', 'U'))
        @eval begin
            # Left division with vectors
            LinearAlgebra.ldiv!(C::DenseCuVector{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::DenseCuVector{T}) where {T<:BlasFloat} =
                sv!('N', $uploc, $isunitc, one(T), parent(A), B, C, 'O')

            # Left division with matrices
            LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                sm!('N', 'N', $uploc, $isunitc, one(T), parent(A), B, C, 'O')

            LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
                sm!('N', 'T', $uploc, $isunitc, one(T), parent(A), parent(B), C, 'O')

            # transb = 'C' is not supported.
            LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                A::$t{T,<:$SparseMatrixType},
                                B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasReal} =
                sm!('N', 'T', $uploc, $isunitc, one(T), parent(A), parent(B), C, 'O')
        end
    end

    ## adjoint/transpose ('uploc' reversed)
    for (t, uploc, isunitc) in ((:LowerTriangular, 'U', 'N'),
                                (:UnitLowerTriangular, 'U', 'U'),
                                (:UpperTriangular, 'L', 'N'),
                                (:UnitUpperTriangular, 'L', 'U'))

        for (opa, transa) in ((:Transpose, 'T'),
                              (:Adjoint, 'C'))
            @eval begin
                # Left division with vectors
                LinearAlgebra.ldiv!(C::DenseCuVector{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuVector{T}) where {T<:BlasFloat} =
                    sv!($transa, $uploc, $isunitc, one(T), parent(parent(A)), B, C, 'O')

                # Left division with matrices
                LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::DenseCuMatrix{T}) where {T<:BlasFloat} =
                    sm!($transa, 'N', $uploc, $isunitc, one(T), parent(parent(A)), B, C, 'O')

                LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::Transpose{T,<:DenseCuMatrix}) where {T<:BlasFloat} =
                    sm!($transa, 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), C, 'O')

                # transb = 'C' is not supported.
                LinearAlgebra.ldiv!(C::DenseCuMatrix{T},
                                    A::$t{T,<:$opa{T,<:$SparseMatrixType}},
                                    B::Adjoint{T,<:DenseCuMatrix}) where {T<:BlasReal} =
                    sm!($transa, 'T', $uploc, $isunitc, one(T), parent(parent(A)), parent(B), C, 'O')
            end
        end
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

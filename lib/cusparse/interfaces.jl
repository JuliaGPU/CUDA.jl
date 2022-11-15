# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal
export _spadjoint, _sptranspose

function _spadjoint(A::CuSparseMatrixCSR)
    Aᴴ = CuSparseMatrixCSC(A.rowPtr, A.colVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSR(Aᴴ)
end
function _sptranspose(A::CuSparseMatrixCSR)
    Aᵀ = CuSparseMatrixCSC(A.rowPtr, A.colVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSR(Aᵀ)
end
function _spadjoint(A::CuSparseMatrixCSC)
    Aᴴ = CuSparseMatrixCSR(A.colPtr, A.rowVal, conj(A.nzVal), reverse(size(A)))
    CuSparseMatrixCSC(Aᴴ)
end
function _sptranspose(A::CuSparseMatrixCSC)
    Aᵀ = CuSparseMatrixCSR(A.colPtr, A.rowVal, A.nzVal, reverse(size(A)))
    CuSparseMatrixCSC(Aᵀ)
end
function _spadjoint(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, conj(A.nzVal), reverse(size(A))..., fmt = :coo)
end
function _sptranspose(A::CuSparseMatrixCOO)
    # we use sparse instead of CuSparseMatrixCOO because we want to sort the matrix.
    sparse(A.colInd, A.rowInd, A.nzVal, reverse(size(A))..., fmt = :coo)
end

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

adjtrans_wrappers = ((identity, identity),
                     (M -> :(Transpose{T, <:$M}), M -> :(_sptranspose(parent($M)))),
                     (M -> :(Adjoint{T, <:$M}), M -> :(_spadjoint(parent($M)))))

op_wrappers = ((identity, T -> 'N', identity),
               (T -> :(Transpose{<:T, <:$T}), T -> 'T', A -> :(parent($A))),
               (T -> :(Adjoint{<:T, <:$T}), T -> T <: Real ? 'T' : 'C', A -> :(parent($A))))

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
        function LinearAlgebra.:(*)(A::$SparseMatrixType{T}, B::$SparseMatrixType{T}) where {T <: BlasFloat}
            CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
            gemm('N', 'N', one(T), A, B, 'O')
        end

        function LinearAlgebra.mul!(C::$SparseMatrixType{T}, A::$SparseMatrixType{T}, B::$SparseMatrixType{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
            CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
            gemm!('N', 'N', alpha, A, B, beta, C, 'O')
        end
    end
end

function LinearAlgebra.:(*)(A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    CuSparseMatrixCOO(A_csr * B_csr)
end

function LinearAlgebra.mul!(C::CuSparseMatrixCOO{T}, A::CuSparseMatrixCOO{T}, B::CuSparseMatrixCOO{T}, alpha::Number, beta::Number) where {T <: BlasFloat}
    CUSPARSE.version() < v"11.1.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    A_csr = CuSparseMatrixCSR(A)
    B_csr = CuSparseMatrixCSR(B)
    C_csr = CuSparseMatrixCSR(C)
    mul!(C_csr, A_csr, B_csr, alpha, beta)
    C = CuSparseMatrixCOO(C_csr)
end

for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        TypeB = wrapb(SparseMatrixType)
        wrapa == identity && wrapb == identity && continue
        @eval begin
            LinearAlgebra.:(*)(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = $(unwrapa(:A)) * $(unwrapb(:B))
            LinearAlgebra.mul!(C::$SparseMatrixType, A::$TypeA, B::$TypeB, alpha::Number, beta::Number) where {T <: BlasFloat} = mul!(C, $(unwrapa(:A)), $(unwrapb(:B)), alpha, beta)
        end
    end
end

for op in (:(+), :(-))
    for (wrapa, unwrapa) in adjtrans_wrappers, (wrapb, unwrapb) in adjtrans_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}))
            TypeA = wrapa(SparseMatrixType)
            TypeB = wrapb(SparseMatrixType)
            @eval Base.$op(A::$TypeA, B::$TypeB) where {T <: BlasFloat} = geam(one(T), $(unwrapa(:A)), $(op)(one(T)), $(unwrapb(:B)), 'O')
        end
    end

    @eval begin
        Base.$op(A::CuSparseVector{T}, B::CuSparseVector{T}) where {T <: BlasFloat} = axpby(one(T), A, $(op)(one(T)), B, 'O')
        Base.$op(A::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}},
                 B::Union{CuSparseMatrixCOO{T}, Transpose{T,<:CuSparseMatrixCOO}, Adjoint{T,<:CuSparseMatrixCOO}}) where {T <: BlasFloat} =
            CuSparseMatrixCOO($(op)(CuSparseMatrixCSR(A), CuSparseMatrixCSR(B)))
    end

    for (wrap1, unwrap1) in adjtrans_wrappers, (wrap2, unwrap2) in adjtrans_wrappers
        for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCOO{T}), :(CuSparseMatrixBSR{T}))
            Type1 = wrap1(:(CuSparseMatrixCSR{T}))
            Type2 = wrap2(SparseMatrixType)
            @eval begin
                Base.$op(A::$Type1, B::$Type2) where {T <: BlasFloat} = $(op)($(unwrap1(:A)), CuSparseMatrixCSR(B))
                Base.$op(A::$Type2, B::$Type1) where {T <: BlasFloat} = $(op)(CuSparseMatrixCSR(A), $(unwrap1(:B)))
            end
        end

        for SparseMatrixType in (:(CuSparseMatrixCOO{T}), :(CuSparseMatrixBSR{T}))
            Type1 = wrap1(:(CuSparseMatrixCSC{T}))
            Type2 = wrap2(SparseMatrixType)
            @eval begin
                Base.$op(A::$Type1, B::$Type2) where {T <: BlasFloat} = $(op)($(unwrap1(:A)), CuSparseMatrixCSC(B))
                Base.$op(A::$Type2, B::$Type1) where {T <: BlasFloat} = $(op)(CuSparseMatrixCSC(A), $(unwrap1(:B)))
            end
        end
    end
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

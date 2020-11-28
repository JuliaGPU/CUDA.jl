# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasComplex, BlasFloat, BlasReal

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix{T}, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    if version() <= v"10.3.1"
        # Generic mm! doesn't support transposed B on CUDA10
        return mm2!(transa, transb, alpha, A, B, beta, C, 'O')
    end
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

tag_wrappers = ((identity, identity),
                (T -> :(HermOrSym{T, <:$T}), A -> :(parent($A))))
op_wrappers = (
    (identity, 'N', identity),
    (T -> :(Transpose{<:T, <:$T}), 'T', A -> :(parent($A))),
    (T -> :(Adjoint{<:T, <:$T}), 'C', A -> :(parent($A)))
)
for (taga, untaga) in tag_wrappers, (wrapa, transa, unwrapa) in op_wrappers
    TypeA = wrapa(taga(:(CuSparseMatrix{T})))

    @eval begin
        function LinearAlgebra.mul!(C::CuVector{T}, A::$TypeA, B::DenseCuVector{T},
                                    alpha::Number, beta::Number) where {T <: BlasFloat}
            mv_wrapper($transa, alpha, $(untaga(unwrapa(:A))), B, beta, C)
        end
    end

    for (tagb, untagb) in tag_wrappers, (wrapb, transb, unwrapb) in op_wrappers
        TypeB = wrapb(tagb(:(DenseCuMatrix{T})))

        isadjoint(expr) = expr.head == :curly && expr.args[1] == :Adjoint
        if isadjoint(TypeA) || isadjoint(TypeB)
            # CUSPARSE defines adjoints only for complex inputs. For real inputs we run
            # adjoints as tranposes so that we can still support the whole API surface of
            # LinearAlgebra.
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasComplex}
                    mm_wrapper($transa, $transb, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end

            transa_real = transa == 'C' ? 'T' : transa
            transb_real = transb == 'C' ? 'T' : transb
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasReal}
                    mm_wrapper($transa_real, $transb_real, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end
        else
            @eval begin
                function LinearAlgebra.mul!(C::CuMatrix{T}, A::$TypeA, B::$TypeB,
                                            alpha::Number, beta::Number) where {T <: BlasFloat}
                    mm_wrapper($transa, $transb, alpha, $(untaga(unwrapa(:A))),
                               $(untagb(unwrapb(:B))), beta, C)
                end
            end
        end
    end
end

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},
          B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,B,'O','O','O')
Base.:(-)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},
          B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')

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

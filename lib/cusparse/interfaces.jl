# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasFloat

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix{T}, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

LinearAlgebra.mul!(C::CuVector{T},A::CuSparseMatrix,B::DenseCuVector,alpha::Number,beta::Number) where {T} = mv_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::CuVector{T},transA::Transpose{<:Any,<:CuSparseMatrix},B::DenseCuVector,alpha::Number,beta::Number) where {T} = mv_wrapper('T',alpha,parent(transA),B,beta,C)
LinearAlgebra.mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::DenseCuVector,alpha::Number,beta::Number) where {T} = mv_wrapper('C',alpha,parent(adjA),B,beta,C)
LinearAlgebra.mul!(C::CuVector{T},A::HermOrSym{T,<:CuSparseMatrix{T}},B::DenseCuVector{T},alpha::Number,beta::Number) where T = mv_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::CuVector{T},transA::Transpose{<:Any, <:HermOrSym{T,<:CuSparseMatrix{T}}},B::DenseCuVector{T},alpha::Number,beta::Number) where {T} = mv_wrapper('T',alpha,parent(transA),B,beta,C)
LinearAlgebra.mul!(C::CuVector{T},adjA::Adjoint{<:Any, <:HermOrSym{T,<:CuSparseMatrix{T}}},B::DenseCuVector{T},alpha::Number,beta::Number) where {T} = mv_wrapper('C',alpha,parent(adjA),B,beta,C)

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    if version() < v"10.3.1" && A isa CuSparseMatrixCSR
        # generic mm! doesn't work on CUDA 10.1 with CSC matrices
        return mm2!(transa, transb, alpha, A, B, beta, C, 'O')
    end
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

LinearAlgebra.mul!(C::CuMatrix{T},A::CuSparseMatrix{T},B::DenseCuMatrix{T},alpha::Number,beta::Number) where {T} = mm_wrapper('N','N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},A::CuSparseMatrix{T},transB::Transpose{<:Any, <:DenseCuMatrix{T}},alpha::Number,beta::Number)  where {T} = mm_wrapper('N','T',alpha,A,parent(transB),beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:CuSparseMatrix{T}},B::DenseCuMatrix{T},alpha::Number,beta::Number)  where {T} = mm_wrapper('T','N',alpha,parent(transA),B,beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:CuSparseMatrix{T}},transB::Transpose{<:Any, <:DenseCuMatrix{T}},alpha::Number,beta::Number) where {T} = mm_wrapper('T','T',alpha,parent(transA),parent(transB),beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},adjA::Adjoint{<:Any, <:CuSparseMatrix{T}},B::DenseCuMatrix{T},alpha::Number,beta::Number)  where {T} = mm_wrapper('C','N',alpha,parent(adjA),B,beta,C)

LinearAlgebra.mul!(C::CuMatrix{T},A::HermOrSym{<:Number, <:CuSparseMatrix},B::DenseCuMatrix,alpha::Number,beta::Number) where {T} = mm_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},B::DenseCuMatrix,alpha::Number,beta::Number) where {T} = mm_wrapper('T',alpha,parent(transA),B,beta,C)
LinearAlgebra.mul!(C::CuMatrix{T},adjA::Adjoint{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},B::DenseCuMatrix,alpha::Number,beta::Number) where {T} = mm_wrapper('C',alpha,parent(adjA),B,beta,C)

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}}, B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::DenseCuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}       = sv2('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::DenseCuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,B,'O','O','O')
Base.:(-)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')

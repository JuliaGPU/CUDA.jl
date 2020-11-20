# interfacing with other packages

using LinearAlgebra
using LinearAlgebra: BlasFloat

function mv_wrapper(transa::SparseChar, alpha::Number, A::CuSparseMatrix{T}, X::DenseCuVector{T},
                    beta::Number, Y::CuVector{T}) where {T}
    mv!(transa, alpha, A, X, beta, Y, 'O')
end

LinearAlgebra.mul!(C::DenseCuVector{T}, A::CuSparseMatrix,
                   B::DenseCuVector, alpha::Number, beta::Number) where {T} =
    mv_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::DenseCuVector{T}, A::Transpose{T,<:CuSparseMatrix},
                   B::DenseCuVector, alpha::Number, beta::Number) where {T} =
    mv_wrapper('T',alpha,parent(A),B,beta,C)
LinearAlgebra.mul!(C::DenseCuVector{T}, A::Adjoint{T,<:CuSparseMatrix},
                   B::DenseCuVector, alpha::Number, beta::Number) where {T} =
    mv_wrapper('C',alpha,parent(A),B,beta,C)
LinearAlgebra.mul!(C::DenseCuVector{T}, A::HermOrSym{T,<:CuSparseMatrix},
                   B::DenseCuVector{T}, alpha::Number, beta::Number) where T =
    mv_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::DenseCuVector{T}, A::Transpose{<:Any, <:HermOrSym{T,<:CuSparseMatrix}},
                   B::DenseCuVector{T}, alpha::Number, beta::Number) where {T} =
    mv_wrapper('T',alpha,parent(A),B,beta,C)
LinearAlgebra.mul!(C::DenseCuVector{T}, A::Adjoint{<:Any, <:HermOrSym{T,<:CuSparseMatrix}},
                   B::DenseCuVector{T}, alpha::Number, beta::Number) where {T} =
    mv_wrapper('C',alpha,parent(A),B,beta,C)

function mm_wrapper(transa::SparseChar, transb::SparseChar, alpha::Number,
                    A::CuSparseMatrix{T}, B::CuMatrix{T}, beta::Number, C::CuMatrix{T}) where {T}
    if version() < v"10.3.1" && A isa CuSparseMatrixCSR
        # generic mm! doesn't work on CUDA 10.1 with CSC matrices
        return mm2!(transa, transb, alpha, A, B, beta, C, 'O')
    end
    mm!(transa, transb, alpha, A, B, beta, C, 'O')
end

LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::CuSparseMatrix{T},
                   B::DenseCuMatrix{T}, alpha::Number, beta::Number) where {T} =
    mm_wrapper('N','N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::CuSparseMatrix{T},
                   B::Transpose{T, <:DenseCuMatrix}, alpha::Number, beta::Number)  where {T} =
    mm_wrapper('N','T',alpha,A,parent(B),beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::Transpose{<:Any, <:CuSparseMatrix{T}},
                   B::DenseCuMatrix{T}, alpha::Number, beta::Number)  where {T} =
    mm_wrapper('T','N',alpha,parent(A),B,beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::Transpose{<:Any, <:CuSparseMatrix{T}},
                   B::Transpose{<:Any, <:DenseCuMatrix{T}}, alpha::Number, beta::Number) where {T} =
    mm_wrapper('T','T',alpha,parent(A),parent(B),beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::Adjoint{<:Any, <:CuSparseMatrix{T}},
                   B::DenseCuMatrix{T}, alpha::Number, beta::Number)  where {T} =
    mm_wrapper('C','N',alpha,parent(A),B,beta,C)

LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::HermOrSym{<:Number, <:CuSparseMatrix},
                   B::DenseCuMatrix, alpha::Number, beta::Number) where {T} =
    mm_wrapper('N',alpha,A,B,beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::Transpose{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},
                   B::DenseCuMatrix, alpha::Number, beta::Number) where {T} =
    mm_wrapper('T',alpha,parent(A),B,beta,C)
LinearAlgebra.mul!(C::DenseCuMatrix{T}, A::Adjoint{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},
                   B::DenseCuMatrix, alpha::Number, beta::Number) where {T} =
    mm_wrapper('C',alpha,parent(A),B,beta,C)

Base.:(\)(A::UpperTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('N',A,B,'O')
Base.:(\)(A::LowerTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('N',A,B,'O')
Base.:(\)(A::UpperTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('T',parent(A),B,'O')
Base.:(\)(A::LowerTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('T',parent(A),B,'O')
Base.:(\)(A::UpperTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('C',parent(A),B,'O')
Base.:(\)(A::LowerTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuMatrix{T}) where {T<:BlasFloat} =
    sm('C',parent(A),B,'O')

# TODO: some metaprogramming to reduce the amount of definitions here

Base.:(\)(A::UpperTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('N', 'U', one(T), parent(A), B,'O')
Base.:(\)(A::LowerTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('N', 'L', one(T), parent(A), B,'O')
Base.:(\)(A::UpperTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('T', 'L', one(T), parent(parent(A)), B, 'O')
Base.:(\)(A::LowerTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('T', 'U', one(T), parent(parent(A)), B, 'O')
Base.:(\)(A::UpperTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('C', 'L', one(T), parent(parent(A)), B, 'O')
Base.:(\)(A::LowerTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('C', 'U', one(T), parent(parent(A)), B, 'O')

Base.:(\)(A::UnitUpperTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('N', 'U', one(T), parent(A), B, 'O', unit_diag=true)
Base.:(\)(A::UnitLowerTriangular{T, <:AbstractCuSparseMatrix{T}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('N', 'L', one(T), parent(A), B, 'O', unit_diag=true)
Base.:(\)(A::UnitUpperTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('T', 'L', one(T), parent(parent(A)), B, 'O', unit_diag=true)
Base.:(\)(A::UnitLowerTriangular{<:Any, <:Transpose{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('T', 'U', one(T), parent(parent(A)), B, 'O', unit_diag=true)
Base.:(\)(A::UnitUpperTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('C', 'L', one(T), parent(parent(A)), B, 'O', unit_diag=true)
Base.:(\)(A::UnitLowerTriangular{<:Any, <:Adjoint{T, <:AbstractCuSparseMatrix{T}}},
          B::DenseCuVector{T}) where {T<:BlasFloat} =
    sv2('C', 'U', one(T), parent(parent(A)), B, 'O', unit_diag=true)

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},
          B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,B,'O','O','O')
Base.:(-)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},
          B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')

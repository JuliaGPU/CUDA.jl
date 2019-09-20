# integration with LinearAlgebra stdlib

using LinearAlgebra
import LinearAlgebra: BlasFloat, mul!

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::CuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}       = sm('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}}, B::CuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}}, B::CuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::CuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::CuMatrix{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sm('C',parent(adjA),B,'O')

mul!(C::CuVector{T},A::CuSparseMatrix,B::CuVector) where {T} = mv!('N',one(T),A,B,zero(T),C,'O')
mul!(C::CuVector{T},transA::Transpose{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('T',one(T),parent(transA),B,zero(T),C,'O')
mul!(C::CuVector{T},adjA::Adjoint{<:Any,<:CuSparseMatrix},B::CuVector) where {T} = mv!('C',one(T),parent(transA),B,zero(T),C,'O')
mul!(C::CuVector{T},A::HermOrSym{T,<:CuSparseMatrix{T}},B::CuVector{T}) where T = mv!('N',one(T),A,B,zero(T),C,'O')
mul!(C::CuVector{T},transA::Transpose{<:Any, <:HermOrSym{T,<:CuSparseMatrix{T}}},B::CuVector{T}) where {T} = mv!('T',one(T),parent(transA),B,zero(T),C,'O')
mul!(C::CuVector{T},adjA::Adjoint{<:Any, <:HermOrSym{T,<:CuSparseMatrix{T}}},B::CuVector{T}) where {T} = mv!('C',one(T),parent(adjA),B,zero(T),C,'O')

mul!(C::CuMatrix{T},A::CuSparseMatrix{T},B::CuMatrix{T}) where {T} = mm2!('N','N',one(T),A,B,zero(T),C,'O')
mul!(C::CuMatrix{T},A::CuSparseMatrix{T},transB::Transpose{<:Any, CuMatrix{T}})  where {T} = mm2!('N','T',one(T),A,parent(transB),zero(T),C,'O')
mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:CuSparseMatrix{T}},B::CuMatrix{T})  where {T} = mm2!('T','N',one(T),parent(transA),B,zero(T),C,'O')
mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:CuSparseMatrix{T}},transB::Transpose{<:Any, CuMatrix{T}}) where {T} = mm2!('T','T',one(T),parent(transA),parent(transB),zero(T),C,'O')
mul!(C::CuMatrix{T},adjA::Adjoint{<:Any, <:CuSparseMatrix{T}},B::CuMatrix{T})  where {T} = mm2!('C','N',one(T),parent(adjA),B,zero(T),C,'O')

mul!(C::CuMatrix{T},A::HermOrSym{<:Number, <:CuSparseMatrix},B::CuMatrix) where {T} = mm!('N',one(T),A,B,zero(T),C,'O')
mul!(C::CuMatrix{T},transA::Transpose{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},B::CuMatrix) where {T} = mm!('T',one(T),parent(transA),B,zero(T),C,'O')
mul!(C::CuMatrix{T},adjA::Adjoint{<:Any, <:HermOrSym{<:Number, <:CuSparseMatrix}},B::CuMatrix) where {T} = mm!('C',one(T),parent(adjA),B,zero(T),C,'O')

Base.:(\)(A::Union{UpperTriangular{T, S},LowerTriangular{T, S}}, B::CuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}       = sv2('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, S}},B::CuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, S}},B::CuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}} = sv2('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, S}},B::CuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, S}},B::CuVector{T}) where {T<:BlasFloat, S<:AbstractCuSparseMatrix{T}}  = sv2('C',parent(adjA),B,'O')
Base.:(\)(A::AbstractTriangular{T,CuSparseMatrixHYB{T}},B::CuVector{T})       where T = sv('N',A,B,'O')
Base.:(\)(transA::Transpose{T, UpperTriangular{T, CuSparseMatrixHYB{T}}},B::CuVector{T}) where {T<:BlasFloat} = sv('T',parent(transA),B,'O')
Base.:(\)(transA::Transpose{T, LowerTriangular{T, CuSparseMatrixHYB{T}}},B::CuVector{T}) where {T<:BlasFloat} = sv('T',parent(transA),B,'O')
Base.:(\)(adjA::Adjoint{T, UpperTriangular{T, CuSparseMatrixHYB{T}}},B::CuVector{T}) where {T<:BlasFloat} = sv('C',parent(adjA),B,'O')
Base.:(\)(adjA::Adjoint{T, LowerTriangular{T, CuSparseMatrixHYB{T}}},B::CuVector{T}) where {T<:BlasFloat} = sv('C',parent(adjA),B,'O')

Base.:(+)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,B,'O','O','O')
Base.:(-)(A::Union{CuSparseMatrixCSR,CuSparseMatrixCSC},B::Union{CuSparseMatrixCSR,CuSparseMatrixCSC}) = geam(A,-one(eltype(A)),B,'O','O','O')

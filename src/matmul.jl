using LinearAlgebra

function generic_matmatmul!(C::AbstractVecOrMat{R}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{S}) where {T,S,R}
    if size(A,2) != size(B,1)
        throw(DimensionMismatch("matrix A has dimensions $(size(A)), matrix B has dimensions $(size(B))"))
    end
    if size(C,1) != size(A,1) || size(C,2) != size(B,2)
        throw(DimensionMismatch("result C has dimensions $(size(C)), needs $((size(A,1),size(B,2)))"))
    end
    if isempty(A) || isempty(B)
        return fill!(C, zero(R))
    end

    function kernel(C, A, B)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y

        if i <= size(A,1) && j <= size(B,2)
            z2 = zero(A[i, 1]*B[1, j] + A[i, 1]*B[1, j])
            Ctmp = convert(promote_type(R, typeof(z2)), z2)
            for k in 1:size(A,2)
                Ctmp += A[i, k]*B[k, j]
            end
            C[i,j] = Ctmp
        end
    end

    @cuda threads=size(C) kernel(C, A, B)

    C
end

LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::CuVecOrMat, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::CuVecOrMat) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Adjoint{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)
LinearAlgebra.mul!(C::CuVecOrMat, A::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}, B::LinearAlgebra.Transpose{<:Any, <:CuVecOrMat}) = generic_matmatmul!(C, A, B)

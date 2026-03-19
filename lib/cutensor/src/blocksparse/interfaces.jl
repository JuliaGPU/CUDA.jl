## For now call contract in ITensor and rely on UnallocatedArrays to make 
## C in a dry-run of the contraction.
# function Base.:(*)(A::CuTensorBS, B::CuTensorBs)
#     tC = promote_type(eltype(A), eltype(B))
#     A_uniqs = [(idx, i) for (idx, i) in enumerate(A.inds) if !(i in B.inds)]
#     B_uniqs = [(idx, i) for (idx, i) in enumerate(B.inds) if !(i in A.inds)]
#     A_sizes = map(x->size(A,x[1]), A_uniqs)
#     B_sizes = map(x->size(B,x[1]), B_uniqs)
#     A_inds = map(x->x[2], A_uniqs)
#     B_inds = map(x->x[2], B_uniqs)
#     C = CuTensor(CUDA.zeros(tC, Dims(vcat(A_sizes, B_sizes))), vcat(A_inds, B_inds))
#     return mul!(C, A, B)
# end


## LinearAlgebra

using LinearAlgebra

function LinearAlgebra.mul!(C::CuTensorBS, A::CuTensorBS, B::CuTensorBS, α::Number, β::Number)
   contract!(α, 
            A, A.inds, CUTENSOR_OP_IDENTITY,
            B, B.inds, CUTENSOR_OP_IDENTITY, 
            β,
            C, C.inds, CUTENSOR_OP_IDENTITY, 
            CUTENSOR_OP_IDENTITY; jit=CUTENSOR_JIT_MODE_DEFAULT)
   return C
end


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
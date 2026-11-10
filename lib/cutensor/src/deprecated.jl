# Deprecated in v6.6: the elementwise operations are named like the other operations
# (contract!, permute!, reduce!), without NVIDIA's `Execute` suffix.

@deprecate elementwise_binary_execute!(alpha::Number,
                                       A::AbstractArray, Ainds::ModeType, opA::cutensorOperator_t,
                                       gamma::Number,
                                       C::AbstractArray, Cinds::ModeType, opC::cutensorOperator_t,
                                       D::AbstractArray, Dinds::ModeType, opAC::cutensorOperator_t;
                                       kwargs...) elementwise_binary!(alpha, A, Ainds, opA, gamma, C, Cinds, opC, D, Dinds, opAC; kwargs...) false
@deprecate elementwise_binary_execute!(plan::CuTensorPlan, alpha::Number, A::AbstractArray,
                                       gamma::Number, C::AbstractArray, D::AbstractArray) elementwise_binary!(plan, alpha, A, gamma, C, D) false

@deprecate elementwise_trinary_execute!(alpha::Number,
                                        A::AbstractArray, Ainds::ModeType, opA::cutensorOperator_t,
                                        beta::Number,
                                        B::AbstractArray, Binds::ModeType, opB::cutensorOperator_t,
                                        gamma::Number,
                                        C::AbstractArray, Cinds::ModeType, opC::cutensorOperator_t,
                                        D::AbstractArray, Dinds::ModeType, opAB::cutensorOperator_t,
                                        opABC::cutensorOperator_t;
                                        kwargs...) elementwise_trinary!(alpha, A, Ainds, opA, beta, B, Binds, opB, gamma, C, Cinds, opC, D, Dinds, opAB, opABC; kwargs...) false
@deprecate elementwise_trinary_execute!(plan::CuTensorPlan, alpha::Number, A::AbstractArray,
                                        beta::Number, B::AbstractArray, gamma::Number,
                                        C::AbstractArray, D::AbstractArray) elementwise_trinary!(plan, alpha, A, beta, B, gamma, C, D) false

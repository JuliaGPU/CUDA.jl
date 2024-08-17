const ModeType = AbstractVector{<:Union{Char, Integer}}

# remove the CUTENSOR_ prefix from some common enums,
# as they're namespaced to the cuTENSOR module anyway.
@enum_without_prefix cutensorOperator_t CUTENSOR_
@enum_without_prefix cutensorWorksizePreference_t CUTENSOR_
@enum_without_prefix cutensorAlgo_t CUTENSOR_
@enum_without_prefix cutensorJitMode_t CUTENSOR_

is_unary(op::cutensorOperator_t) =  (op ∈ (OP_IDENTITY, OP_SQRT, OP_RELU, OP_CONJ, OP_RCP))
is_binary(op::cutensorOperator_t) = (op ∈ (OP_ADD, OP_MUL, OP_MAX, OP_MIN))

function elementwise_trinary_execute!(
        @nospecialize(alpha::Number),
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(B::AbstractArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::AbstractArray), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_elementwise_trinary(A, Ainds, opA,
                                 B, Binds, opB,
                                 C, Cinds, opC,
                                 D, Dinds, opAB, opABC;
                                 workspace, algo, compute_type)
    else
        plan
    end

    elementwise_trinary_execute!(actual_plan, alpha, A, beta, B, gamma, C, D)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return D
end

function elementwise_trinary_execute!(plan::CuTensorPlan,
                                      @nospecialize(alpha::Number),
                                      @nospecialize(A::AbstractArray),
                                      @nospecialize(beta::Number),
                                      @nospecialize(B::AbstractArray),
                                      @nospecialize(gamma::Number),
                                      @nospecialize(C::AbstractArray),
                                      @nospecialize(D::AbstractArray))
    scalar_type = plan.scalar_type
    cutensorElementwiseTrinaryExecute(handle(), plan,
                                      Ref{scalar_type}(alpha), A,
                                      Ref{scalar_type}(beta), B,
                                      Ref{scalar_type}(gamma), C, D,
                                      stream())
    return D
end

function plan_elementwise_trinary(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::AbstractArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::AbstractArray), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAB)  && throw(ArgumentError("opAB must be a binary op!"))
    !is_binary(opABC) && throw(ArgumentError("opABC must be a binary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    descC = CuTensorDescriptor(C)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    modeD = modeC

    actual_compute_type = if compute_type === nothing
        elementwise_trinary_compute_types[(eltype(A), eltype(B), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateElementwiseTrinary(handle(),
                                     desc,
                                     descA, modeA, opA,
                                     descB, modeB, opB,
                                     descC, modeC, opC,
                                     descD, modeD,
                                     opAB, opABC,
                                     actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function elementwise_binary_execute!(
        @nospecialize(alpha::Number),
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::AbstractArray), Dinds::ModeType, opAC::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        plan::Union{CuTensorPlan, Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_elementwise_binary(A, Ainds, opA,
                                C, Cinds, opC,
                                D, Dinds, opAC;
                                workspace, algo, compute_type)
    else
        plan
    end

    elementwise_binary_execute!(actual_plan, alpha, A, gamma, C, D)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return D
end

function elementwise_binary_execute!(plan::CuTensorPlan,
                                     @nospecialize(alpha::Number),
                                     @nospecialize(A::AbstractArray),
                                     @nospecialize(gamma::Number),
                                     @nospecialize(C::AbstractArray),
                                     @nospecialize(D::AbstractArray))
    scalar_type = plan.scalar_type
    cutensorElementwiseBinaryExecute(handle(), plan,
                                     Ref{scalar_type}(alpha), A,
                                     Ref{scalar_type}(gamma), C, D,
                                     stream())
    return D
end

function plan_elementwise_binary(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::AbstractArray), Dinds::ModeType, opAC::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=eltype(C))
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC

    actual_compute_type = if compute_type === nothing
        elementwise_binary_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateElementwiseBinary(handle(),
                                     desc,
                                     descA, modeA, opA,
                                     descC, modeC, opC,
                                     descD, modeD,
                                     opAC,
                                     actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function permute!(
        @nospecialize(alpha::Number),
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::AbstractArray), Binds::ModeType;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        plan::Union{CuTensorPlan, Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_permutation(A, Ainds, opA,
                         B, Binds;
                         workspace, algo, compute_type)
    else
        plan
    end

    permute!(actual_plan, alpha, A, B)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return B
end

function permute!(plan::CuTensorPlan,
                  @nospecialize(alpha::Number),
                  @nospecialize(A::AbstractArray),
                  @nospecialize(B::AbstractArray))
    scalar_type = plan.scalar_type
    cutensorPermute(handle(), plan,
                    Ref{scalar_type}(alpha), A, B,
                    stream())
    return B
end

function plan_permutation(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::AbstractArray), Binds::ModeType;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing)
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)

    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)

    actual_compute_type = if compute_type === nothing
        permutation_compute_types[(eltype(A), eltype(B))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreatePermutation(handle(), desc,
                              descA, modeA, opA,
                              descB, modeB,
                              actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function contract!(
        @nospecialize(alpha::Number),
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::AbstractArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        plan::Union{CuTensorPlan, Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut;
                         jit, workspace, algo, compute_type)
    else
        plan
    end

    contract!(actual_plan, alpha, A, B, beta, C)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function contract!(plan::CuTensorPlan,
                   @nospecialize(alpha::Number),
                   @nospecialize(A::AbstractArray),
                   @nospecialize(B::AbstractArray),
                   @nospecialize(beta::Number),
                   @nospecialize(C::AbstractArray))
    scalar_type = plan.scalar_type
    cutensorContract(handle(), plan,
                     Ref{scalar_type}(alpha), A, B,
                     Ref{scalar_type}(beta), C, C,
                     plan.workspace, sizeof(plan.workspace), stream())
    return C
end

function plan_contraction(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::AbstractArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        contraction_compute_types[(eltype(A), eltype(B), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateContraction(handle(),
                              desc,
                              descA, modeA, opA,
                              descB, modeB, opB,
                              descC, modeC, opC,
                              descC, modeC,
                              actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function reduce!(
        @nospecialize(alpha::Number),
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        plan::Union{CuTensorPlan, Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_reduction(A, Ainds, opA, C, Cinds, opC, opReduce; workspace, algo, compute_type)
    else
        plan
    end

    reduce!(actual_plan, alpha, A, beta, C)

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function reduce!(plan::CuTensorPlan,
                 @nospecialize(alpha::Number),
                 @nospecialize(A::AbstractArray),
                 @nospecialize(beta::Number),
                 @nospecialize(C::AbstractArray))
    scalar_type = plan.scalar_type
    cutensorReduce(handle(), plan,
                   Ref{scalar_type}(alpha), A,
                   Ref{scalar_type}(beta), C, C,
                   plan.workspace, sizeof(plan.workspace), stream())
    return C
end

function plan_reduction(
        @nospecialize(A::AbstractArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::AbstractArray), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce)  && throw(ArgumentError("opReduce must be a binary op!"))
    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)

    actual_compute_type = if compute_type === nothing
        reduction_compute_types[(eltype(A), eltype(C))]
    else
        compute_type
    end

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateReduction(handle(),
                            desc,
                            descA, modeA, opA,
                            descC, modeC, opC,
                            descC, modeC, opReduce,
                            actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

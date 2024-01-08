export elementwise_binary!, elementwise_trinary!,
       permutation!, contraction!, reduction!

const ModeType = AbstractVector{<:Union{Char, Integer}}

# remove the CUTENSOR_ prefix from some common enums,
# as they're namespaced to the cuTENSOR module anyway.
@enum_without_prefix cutensorOperator_t CUTENSOR_
@enum_without_prefix cutensorWorksizePreference_t CUTENSOR_
@enum_without_prefix cutensorAlgo_t CUTENSOR_
@enum_without_prefix cutensorJitMode_t CUTENSOR_

is_unary(op::cutensorOperator_t) =  (op ∈ (OP_IDENTITY, OP_SQRT, OP_RELU, OP_CONJ, OP_RCP))
is_binary(op::cutensorOperator_t) = (op ∈ (OP_ADD, OP_MUL, OP_MAX, OP_MIN))

function elementwise_trinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(B::DenseCuArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C),
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

    scalar_type = eltype(A)
    cutensorElementwiseTrinaryExecute(handle(), actual_plan,
                                      Ref{scalar_type}(alpha), A,
                                      Ref{scalar_type}(beta), B,
                                      Ref{scalar_type}(gamma), C, D,
                                      stream())

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return D
end

function plan_elementwise_trinary(
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::DenseCuArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C))
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

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateElementwiseTrinary(handle(),
                                     desc,
                                     descA, modeA, opA,
                                     descB, modeB, opB,
                                     descC, modeC, opC,
                                     descD, modeD,
                                     opAB, opABC,
                                     compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function elementwise_binary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAC::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C),
        plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_elementwise_binary(A, Ainds, opA,
                                C, Cinds, opC,
                                D, Dinds, opAC;
                                workspace, algo, compute_type)
    else
        plan
    end

    scalar_type = eltype(A)
    cutensorElementwiseBinaryExecute(handle(), actual_plan,
                                     Ref{scalar_type}(alpha), A,
                                     Ref{scalar_type}(gamma), C, D,
                                     stream())

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return D
end

function plan_elementwise_binary(
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAC::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C))
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

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateElementwiseBinary(handle(),
                                     desc,
                                     descA, modeA, opA,
                                     descC, modeC, opC,
                                     descD, modeD,
                                     opAC,
                                     compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function permutation!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::DenseCuArray), Binds::ModeType;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(B),
        plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_permutation(A, Ainds, opA,
                         B, Binds;
                         workspace, algo, compute_type)
    else
        plan
    end

    scalar_type = eltype(B)
    cutensorPermute(handle(), actual_plan,
                    Ref{scalar_type}(alpha), A, B,
                    stream())

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return B
end

function plan_permutation(
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::DenseCuArray), Binds::ModeType;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(B))
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)

    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreatePermutation(handle(), desc,
                              descA, modeA, opA,
                              descB, modeB,
                              compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function contraction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::DenseCuArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C),
        plan::Union{CuTensorPlan, Nothing}=nothing)

    # XXX: save these as parameters of the plan?
    actual_plan = if plan === nothing
        plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut;
                         jit, workspace, algo, compute_type)
    else
        plan
    end

    output_type = eltype(C)
    scalar_type = scalar_types[(output_type, compute_type)]
    cutensorContract(handle(), actual_plan,
                     Ref{scalar_type}(alpha), A, B,
                     Ref{scalar_type}(beta),  C, C,
                     actual_plan.workspace, sizeof(actual_plan.workspace), stream())

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function plan_contraction(
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::DenseCuArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C))
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    output_type = eltype(C)
    scalar_type = scalar_types[(output_type, compute_type)]
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateContraction(handle(),
                              desc,
                              descA, modeA, opA,
                              descB, modeB, opB,
                              descC, modeC, opC,
                              descC, modeC,
                              compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

function reduction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C),
        plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_reduction(A, Ainds, opA, C, Cinds, opC, opReduce; workspace, algo, compute_type)
    else
        plan
    end

    output_type = eltype(C)
    scalar_type = scalar_types[(output_type, compute_type)]
    cutensorReduce(handle(), actual_plan,
                   Ref{scalar_type}(alpha), A,
                   Ref{scalar_type}(beta),  C, C,
                   actual_plan.workspace, sizeof(actual_plan.workspace), stream())

    if plan === nothing
        CUDA.unsafe_free!(actual_plan)
    end

    return C
end

function plan_reduction(
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT, compute_type::Type=eltype(C))
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce)  && throw(ArgumentError("opReduce must be a binary op!"))
    descA = CuTensorDescriptor(A)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateReduction(handle(),
                            desc,
                            descA, modeA, opA,
                            descC, modeC, opC,
                            descC, modeC, opReduce,
                            compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
end

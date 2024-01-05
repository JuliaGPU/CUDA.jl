const ModeType = AbstractVector{<:Union{Char, Integer}}

is_unary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_SQRT, CUTENSOR_OP_RELU, CUTENSOR_OP_CONJ,
            CUTENSOR_OP_RCP))
is_binary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_ADD, CUTENSOR_OP_MUL, CUTENSOR_OP_MAX, CUTENSOR_OP_MIN))

function elementwiseTrinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Union{Array, DenseCuArray}), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C), plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_elementwiseTrinary(A, Ainds, opA,
                                B, Binds, opB,
                                C, Cinds, opC,
                                D, Dinds, opAB, opABC;
                                ws_pref, algo, compute_type)
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

function plan_elementwiseTrinary(
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Union{Array, DenseCuArray}), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C))
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
    cutensorCreatePlanPreference(handle(), plan_pref, algo, CUTENSOR_JIT_MODE_NONE)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=ws_pref)
end

function elementwiseBinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Union{Array, DenseCuArray}), Dinds::ModeType, opAC::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C), plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_elementwiseBinary(A, Ainds, opA,
                               C, Cinds, opC,
                               D, Dinds, opAC;
                               ws_pref, algo, compute_type)
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

function plan_elementwiseBinary(
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Union{Array, DenseCuArray}), Dinds::ModeType, opAC::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C))
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
    cutensorCreatePlanPreference(handle(), plan_pref, algo, CUTENSOR_JIT_MODE_NONE)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=ws_pref)
end

function permutation!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(B), plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_permutation(A, Ainds, opA,
                         B, Binds;
                         ws_pref, algo, compute_type)
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
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(B))
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
    cutensorCreatePlanPreference(handle(), plan_pref, algo, CUTENSOR_JIT_MODE_NONE)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=ws_pref)
end

function contraction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C), plan::Union{CuTensorPlan, Nothing}=nothing)

    # XXX: save these as parameters of the plan?
    actual_plan = if plan === nothing
        plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut; ws_pref, algo, compute_type)
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
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, DenseCuArray}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C))
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
    cutensorCreatePlanPreference(handle(), plan_pref, algo, CUTENSOR_JIT_MODE_NONE)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=ws_pref)
end

function reduction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C), plan::Union{CuTensorPlan, Nothing}=nothing)

    actual_plan = if plan === nothing
        plan_reduction(A, Ainds, opA, C, Cinds, opC, opReduce; ws_pref, algo, compute_type)
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
        @nospecialize(A::Union{Array, DenseCuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(C::Union{Array, DenseCuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t;
        ws_pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C))
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
    cutensorCreatePlanPreference(handle(), plan_pref, algo, CUTENSOR_JIT_MODE_NONE)

    CuTensorPlan(desc[], plan_pref[]; workspacePref=ws_pref)
end

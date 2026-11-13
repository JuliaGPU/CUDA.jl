function contract!(
        @nospecialize(alpha::Number),
        A::CuTensorBS, Ainds::ModeType, opA::cutensorOperator_t,
        B::CuTensorBS, Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        C::CuTensorBS, Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        reproducible::Bool=false,
        plan::Union{CuTensorPlan, Nothing}=nothing)
    actual_plan = if plan === nothing
        plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut;
                         jit, workspace, algo, compute_type, reproducible)
    else
        plan
    end

    contract!(actual_plan, alpha, A, B, beta, C)

    if plan === nothing
        CUDACore.unsafe_free!(actual_plan)
    end

    return C
end

function contract!(plan::CuTensorPlan,
                   @nospecialize(alpha::Number),
                   A::CuTensorBS,
                   B::CuTensorBS,
                   @nospecialize(beta::Number),
                   C::CuTensorBS)
    scalar_type = plan.scalar_type

    # cuTENSOR takes a host array with a device pointer per nonzero block
    A_ptrs = CuPtr{Cvoid}[pointer(block) for block in A.nonzero_data]
    B_ptrs = CuPtr{Cvoid}[pointer(block) for block in B.nonzero_data]
    C_ptrs = CuPtr{Cvoid}[pointer(block) for block in C.nonzero_data]

    cutensorBlockSparseContract(handle(), plan,
                                Ref{scalar_type}(alpha), A_ptrs, B_ptrs,
                                Ref{scalar_type}(beta), C_ptrs, C_ptrs,
                                plan.workspace, sizeof(plan.workspace), stream())
    return C
end

function plan_contraction(
        A::CuTensorBS, Ainds::ModeType, opA::cutensorOperator_t,
        B::CuTensorBS, Binds::ModeType, opB::cutensorOperator_t,
        C::CuTensorBS, Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing,
        reproducible::Bool=false)
    modeA, modeB, modeC, actual_compute_type =
        contraction_arguments(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut, compute_type)
    descA = CuTensorBSDescriptor(A)
    descB = CuTensorBSDescriptor(B)
    descC = CuTensorBSDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)

    desc = Ref{cutensorOperationDescriptor_t}()
    cutensorCreateBlockSparseContraction(handle(),
                                         desc,
                                         descA, modeA, opA,
                                         descB, modeB, opB,
                                         descC, modeC, opC,
                                         descC, modeC,
                                         actual_compute_type)

    if reproducible
        cutensorOperationDescriptorSetAttribute(handle(), desc[],
            CUTENSOR_OPERATION_DESCRIPTOR_BLOCKSPARSE_REPRODUCIBLE,
            Ref{Int32}(1), sizeof(Int32))
    end

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

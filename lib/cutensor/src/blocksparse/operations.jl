function nonzero_blocks(A::CuTensorBS)
    return A.nonzero_data
end

function contract!(
        @nospecialize(alpha::Number),
        @nospecialize(A), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C), Cinds::ModeType, opC::cutensorOperator_t,
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

    contractBS!(actual_plan, alpha, nonzero_blocks(A), nonzero_blocks(B), beta, nonzero_blocks(C))
    
    if plan === nothing
    CUDACore.unsafe_free!(actual_plan)
    end

    return C
end

## This function assumes A, B, and C are Arrays of pointers to CuArrays.
## Please overwrite the `nonzero_blocks` function for your datatype to access this function from contract!
function contractBS!(plan::CuTensorPlan,
                   @nospecialize(alpha::Number),
                   @nospecialize(A::AbstractArray),
                   @nospecialize(B::AbstractArray),
                   @nospecialize(beta::Number),
                   @nospecialize(C::AbstractArray))
    scalar_type = plan.scalar_type

    # Extract GPU pointers from each CuArray block
    # cuTENSOR expects a host-accessible array of GPU pointers
    A_ptrs = CuPtr{Cvoid}[pointer(block) for block in A]
    B_ptrs = CuPtr{Cvoid}[pointer(block) for block in B]
    C_ptrs = CuPtr{Cvoid}[pointer(block) for block in C]
    
    cutensorBlockSparseContract(handle(), plan, 
                                            Ref{scalar_type}(alpha), A_ptrs, B_ptrs, 
                                            Ref{scalar_type}(beta),  C_ptrs, C_ptrs, 
                                            plan.workspace, sizeof(plan.workspace), stream())
    return C
end

function plan_contraction(
        @nospecialize(A), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        jit::cutensorJitMode_t=JIT_MODE_NONE,
        workspace::cutensorWorksizePreference_t=WORKSPACE_DEFAULT,
        algo::cutensorAlgo_t=ALGO_DEFAULT,
        compute_type::Union{DataType, cutensorComputeDescriptorEnum, Nothing}=nothing)

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    
    descA = CuTensorBSDescriptor(A)
    descB = CuTensorBSDescriptor(B)
    descC = CuTensorBSDescriptor(C)
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
    cutensorCreateBlockSparseContraction(handle(),
    desc, 
    descA, modeA, opA,
    descB, modeB, opB,
    descC, modeC, opC,
    descC, modeC, actual_compute_type)

    plan_pref = Ref{cutensorPlanPreference_t}()
    cutensorCreatePlanPreference(handle(), plan_pref, algo, jit)

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

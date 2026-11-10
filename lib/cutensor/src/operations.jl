const ModeType = AbstractVector{<:Union{Char, Integer}}

# remove the CUTENSOR_ prefix from some common enums,
# as they're namespaced to the cuTENSOR module anyway.
@enum_without_prefix visibility=:public cutensorOperator_t CUTENSOR_
@enum_without_prefix visibility=:public cutensorWorksizePreference_t CUTENSOR_
@enum_without_prefix visibility=:public cutensorAlgo_t CUTENSOR_
@enum_without_prefix visibility=:public cutensorJitMode_t CUTENSOR_

@public contract!, permute!, reduce!, elementwise_binary!, elementwise_trinary!,
        plan_contraction, plan_permutation, plan_reduction,
        plan_elementwise_binary, plan_elementwise_trinary

is_unary(op::cutensorOperator_t) =  (op ∈ (OP_IDENTITY, OP_SQRT, OP_RELU, OP_CONJ, OP_RCP))
is_binary(op::cutensorOperator_t) = (op ∈ (OP_ADD, OP_MUL, OP_MAX, OP_MIN))

"""
    elementwise_trinary!(alpha, A, Ainds, opA, beta, B, Binds, opB, gamma, C, Cinds, opC,
                         D, Dinds, opAB, opABC;
                         workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                         compute_type=nothing, plan=nothing) -> D
    elementwise_trinary!(plan::CuTensorPlan, alpha, A, beta, B, gamma, C, D) -> D

Combine three tensors elementwise. Computes
`D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C))`, where `A` and `B` are
permuted to match `C`'s modes, and returns `D`. `opA`, `opB` and `opC` are unary operators,
`opAB` and `opABC` binary ones (`OP_ADD`, `OP_MUL`, `OP_MAX`, `OP_MIN`). `D` must have the
same size, strides and element type as `C`, and may be `C` itself; it takes `C`'s modes,
so `Dinds` is not used. See [`contract!`](@ref) for the modes, keyword arguments and the
second form, which takes a plan from [`plan_elementwise_trinary`](@ref).

Wraps `cutensorCreateElementwiseTrinary` and `cutensorElementwiseTrinaryExecute`.
"""
function elementwise_trinary!(
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

    elementwise_trinary!(actual_plan, alpha, A, beta, B, gamma, C, D)

    if plan === nothing
        CUDACore.unsafe_free!(actual_plan)
    end

    return D
end

function elementwise_trinary!(plan::CuTensorPlan,
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

"""
    plan_elementwise_trinary(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, D, Dinds, opAB, opABC;
                             jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT,
                             algo=ALGO_DEFAULT, compute_type=nothing) -> CuTensorPlan

Plan the operation of [`elementwise_trinary!`](@ref) for arrays with the sizes, strides,
element types and modes of `A`, `B`, `C` and `D`, to execute it several times. Wraps
`cutensorCreateElementwiseTrinary` and `cutensorCreatePlan`.
"""
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

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

"""
    elementwise_binary!(alpha, A, Ainds, opA, gamma, C, Cinds, opC, D, Dinds, opAC;
                        workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                        compute_type=nothing, plan=nothing) -> D
    elementwise_binary!(plan::CuTensorPlan, alpha, A, gamma, C, D) -> D

Combine two tensors elementwise. Computes `D = opAC(alpha * opA(A), gamma * opC(C))`,
where `A` is permuted to match `C`'s modes, and returns `D`. `opA` and `opC` are unary
operators, `opAC` is a binary one (`OP_ADD`, `OP_MUL`, `OP_MAX`, `OP_MIN`). `D` must have
the same size, strides and element type as `C`, and may be `C` itself; it takes `C`'s
modes, so `Dinds` is not used. See [`contract!`](@ref) for the modes, keyword arguments
and the second form, which takes a plan from [`plan_elementwise_binary`](@ref).

Wraps `cutensorCreateElementwiseBinary` and `cutensorElementwiseBinaryExecute`.
"""
function elementwise_binary!(
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

    elementwise_binary!(actual_plan, alpha, A, gamma, C, D)

    if plan === nothing
        CUDACore.unsafe_free!(actual_plan)
    end

    return D
end

function elementwise_binary!(plan::CuTensorPlan,
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

"""
    plan_elementwise_binary(A, Ainds, opA, C, Cinds, opC, D, Dinds, opAC;
                            jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT,
                            algo=ALGO_DEFAULT, compute_type=eltype(C)) -> CuTensorPlan

Plan the operation of [`elementwise_binary!`](@ref) for arrays with the sizes, strides,
element types and modes of `A`, `C` and `D`, to execute it several times. Wraps
`cutensorCreateElementwiseBinary` and `cutensorCreatePlan`.
"""
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

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

"""
    cuTENSOR.permute!(alpha, A, Ainds, opA, B, Binds;
                      workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                      compute_type=nothing, plan=nothing) -> B
    cuTENSOR.permute!(plan::CuTensorPlan, alpha, A, B) -> B

Permute the dimensions of a tensor. Computes `B = alpha * opA(A)`, with `A`'s dimensions
reordered from the modes `Ainds` to the modes `Binds`, and returns `B`. `opA` is a unary
operator. See [`contract!`](@ref) for the modes, keyword arguments and the second form,
which takes a plan from [`plan_permutation`](@ref).

This function is not `Base.permute!`, which permutes a vector; call it qualified.

Wraps `cutensorCreatePermutation` and `cutensorPermute`.
"""
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
        CUDACore.unsafe_free!(actual_plan)
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

"""
    plan_permutation(A, Ainds, opA, B, Binds;
                     jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                     compute_type=nothing) -> CuTensorPlan

Plan the operation of [`permute!`](@ref) for arrays with the sizes, strides, element types
and modes of `A` and `B`, to execute it several times. Wraps
`cutensorCreatePermutation` and `cutensorCreatePlan`.
"""
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

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

"""
    contract!(alpha, A, Ainds, opA, B, Binds, opB, beta, C, Cinds, opC, opOut;
              jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
              compute_type=nothing, plan=nothing) -> C
    contract!(plan::CuTensorPlan, alpha, A, B, beta, C) -> C

Contract two tensors. Computes `C = alpha * opA(A) * opB(B) + beta * C`, summing over the
modes that `A` and `B` share but `C` does not, and returns `C`.

The modes `Ainds`, `Binds` and `Cinds` label the dimensions of each array, one `Char` or
integer per dimension; dimensions with the same label are matched. The arrays are dense
`CuArray`s. `opA` and `opB` are unary operators applied to the elements (`OP_IDENTITY`,
`OP_CONJ`, `OP_SQRT`, `OP_RELU` or `OP_RCP`). `opC` must be `OP_IDENTITY`, and `opOut`
must be a unary operator but is not used.

The keyword arguments, shared by the other operations, are:

- `jit`: `JIT_MODE_DEFAULT` to compile a kernel for this operation, which can be faster
  on recent GPUs (see [`write_cache!`](@ref)), or `JIT_MODE_NONE`. Only `contract!` and
  the `plan_*` functions take it;
- `workspace`: how much workspace memory cuTENSOR may use (`WORKSPACE_MIN`,
  `WORKSPACE_DEFAULT` or `WORKSPACE_MAX`);
- `algo`: the algorithm (`ALGO_DEFAULT`, `ALGO_DEFAULT_PATIENT`, `ALGO_GETT`,
  `ALGO_TGETT` or `ALGO_TTGT`);
- `compute_type`: the precision to compute in, as a type (`Float16`, `Float32` or
  `Float64`) or a [`cutensorComputeDescriptorEnum`](@ref) value such as
  `COMPUTE_DESC_TF32`. By default it follows from the element types;
- `plan`: a [`CuTensorPlan`](@ref) from [`plan_contraction`](@ref) to use instead of
  making a new one. Without it, a plan is made and freed for every call.

The second form executes a plan made by [`plan_contraction`](@ref) on arrays with the
sizes, strides and element types it was planned for, so that repeated contractions don't
pay for planning. It does not free the plan.

`contract!` also contracts block-sparse tensors ([`CuTensorBS`](@ref)), with an extra
`reproducible::Bool` keyword argument; this is experimental.

Wraps `cutensorCreateContraction` and `cutensorContract`.
"""
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
        CUDACore.unsafe_free!(actual_plan)
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

"""
    plan_contraction(A, Ainds, opA, B, Binds, opB, C, Cinds, opC, opOut;
                     jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                     compute_type=nothing) -> CuTensorPlan

Plan the contraction of [`contract!`](@ref) for arrays with the sizes, strides, element
types and modes of `A`, `B` and `C`, to execute it several times. Wraps
`cutensorCreateContraction` and `cutensorCreatePlan`.
"""
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
    opC != CUTENSOR_OP_IDENTITY && throw(ArgumentError("opC must be the identity op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    descC = CuTensorDescriptor(C)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    modeA = collect(Cint, Ainds)
    length(modeA) == ndims(A) || throw(ArgumentError("Ainds must match number of dimensions in A!"))
    modeB = collect(Cint, Binds)
    length(modeB) == ndims(B) || throw(ArgumentError("Binds must match number of dimensions in B!"))
    modeC = collect(Cint, Cinds)
    length(modeC) == ndims(C) || throw(ArgumentError("Cinds must match number of dimensions in C!"))

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

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

"""
    reduce!(alpha, A, Ainds, opA, beta, C, Cinds, opC, opReduce;
            workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
            compute_type=nothing, plan=nothing) -> C
    reduce!(plan::CuTensorPlan, alpha, A, beta, C) -> C

Reduce a tensor over some of its dimensions. Computes
`C = alpha * opReduce(opA(A)) + beta * opC(C)`, where the binary operator `opReduce`
(`OP_ADD`, `OP_MUL`, `OP_MAX`, `OP_MIN`) combines the elements along the modes of `A` that
are not in `Cinds`, and returns `C`. `opA` and `opC` are unary operators. See
[`contract!`](@ref) for the modes, keyword arguments and the second form, which takes a
plan from [`plan_reduction`](@ref).

Wraps `cutensorCreateReduction` and `cutensorReduce`.
"""
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
        CUDACore.unsafe_free!(actual_plan)
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

"""
    plan_reduction(A, Ainds, opA, C, Cinds, opC, opReduce;
                   jit=JIT_MODE_NONE, workspace=WORKSPACE_DEFAULT, algo=ALGO_DEFAULT,
                   compute_type=nothing) -> CuTensorPlan

Plan the reduction of [`reduce!`](@ref) for arrays with the sizes, strides, element types
and modes of `A` and `C`, to execute it several times. Wraps
`cutensorCreateReduction` and `cutensorCreatePlan`.
"""
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

    plan = CuTensorPlan(desc[], plan_pref[]; workspacePref=workspace)
    cutensorDestroyOperationDescriptor(desc[])
    cutensorDestroyPlanPreference(plan_pref[])
    return plan
end

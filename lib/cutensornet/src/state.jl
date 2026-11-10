# the state API, for simulating quantum circuits and other tensor network states.
#
# state modes are 0-based, and tensor operators act on them following NVIDIA's conventions:
# with the default layout, the elements of an operator applied to a state are stored in the
# row-following order of its textbook definition. the tensors of a `CuNetworkOperator` are
# different: their modes are the state modes they act on, in order, followed by the same for
# the dual modes, so multi-mode tensors aren't laid out like the ones applied to states.
#
# all data handed to the library must stay alive for as long as the object it was given to,
# which these wrappers take care of. that data must be device memory: host pointers only work
# on systems where the GPU can access all of host memory (e.g. with HMM), and cause illegal
# memory accesses elsewhere.

const DEFAULT_MAX_WORKSPACE = 2^30

# the library requires objects to be destroyed before the ones they were created from, but
# finalizers run in no particular order. those objects are therefore counted as being in use,
# and what's finalized while still in use gets destroyed when its last user is.
const usage_lock = ReentrantLock()

function retain!(owner)
    Base.@lock usage_lock owner.users += 1
    return owner
end

function release!(owner)
    Base.@lock usage_lock begin
        owner.users -= 1
        owner.users == 0 && owner.retired && owner.destroy(owner)
    end
    return
end

function retire!(owner)
    Base.@lock usage_lock begin
        owner.retired = true
        owner.users == 0 && owner.destroy(owner)
    end
    return
end

@public CuNetworkOperator, append_product!, append_mpo!, CuNetworkState, apply!,
        apply_with_gradient!, update_gradient!, apply_diagonal!, apply_controlled!, update!,
        apply_unitary_channel!, apply_general_channel!, initialize_mps!, finalize_mps!,
        capture_mps!, details, CuStateAccessor, CuStateExpectation, CuStateMarginal,
        CuStateSampler, CuStateProjectionMPS, configure!, info, prepare!, compute!,
        compute_with_gradients!, flops, marginal_kind, sample!, tensor_info, compute_env!,
        extract_tensor!, insert_tensor!, update_coefficient!, update_dual_tensors!

"""
    CuNetworkOperator{T}(extents)

An operator on a state with modes of the given extents, as a sum of terms that are added with
[`append_product!`](@ref) and [`append_mpo!`](@ref). Apply it to a [`CuNetworkState`](@ref)
with [`apply!`](@ref), or compute its expectation value with
[`CuStateExpectation`](@ref).

The tensors of an operator have the state modes they act on as their first modes, in order,
followed by the corresponding dual modes. This differs from the layout of the operators
that [`apply!`](@ref) takes.

Wraps `cutensornetNetworkOperator_t`.
"""
mutable struct CuNetworkOperator{T}
    handle::cutensornetNetworkOperator_t
    tensors::Vector{Any}
    users::Int
    retired::Bool
    destroy::Function
    function CuNetworkOperator{T}(extents) where {T}
        op_ref = Ref{cutensornetNetworkOperator_t}()
        cutensornetCreateNetworkOperator(handle(), length(extents), collect(Int64, extents), T, op_ref)
        obj = new{T}(op_ref[], Any[], 0, false, cutensornetDestroyNetworkOperator)
        resource_finalizer(retire!, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetNetworkOperator_t}, op::CuNetworkOperator) = op.handle

"""
    append_product!(op::CuNetworkOperator{T}, coefficient, modes::Vector{<:Vector{<:Integer}},
                    tensors::Vector{<:CuArray{T}}) -> Int64

Add a term to the operator: `coefficient` times the product of `tensors`, each acting on
the (0-based) state modes in the corresponding element of `modes`. Returns the id of the
term.

Wraps `cutensornetNetworkOperatorAppendProduct`.
"""
function append_product!(op::CuNetworkOperator{T}, coefficient::Number, modes::Vector{<:Vector{<:Integer}}, tensors::Vector{<:CuArray{T}}) where {T}
    length(modes) == length(tensors) || throw(DimensionMismatch("need as many tensors as sets of modes."))
    id = Ref{Int64}()
    cutensornetNetworkOperatorAppendProduct(handle(), op, ComplexF64(coefficient), length(tensors), Int32.(length.(modes)),
                                            [convert(Vector{Int32}, m) for m in modes], C_NULL, pointer.(tensors), id)
    push!(op.tensors, tensors)
    return id[]
end

"""
    append_mpo!(op::CuNetworkOperator{T}, coefficient, modes::Vector{<:Integer},
                tensors::Vector{<:CuArray{T}}; boundary_condition=CUTENSORNET_BOUNDARY_CONDITION_OPEN)
        -> Int64

Add a term to the operator: `coefficient` times a matrix product operator acting on the
(0-based) state `modes`, with one tensor per mode. Each tensor has the modes (left bond,
state mode, right bond, dual state mode), without the outer bonds at the boundaries of an
open chain. Returns the id of the term.

Wraps `cutensornetNetworkOperatorAppendMPO`.
"""
function append_mpo!(op::CuNetworkOperator{T}, coefficient::Number, modes::Vector{<:Integer}, tensors::Vector{<:CuArray{T}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    length(modes) == length(tensors) || throw(DimensionMismatch("need one tensor for each mode."))
    id = Ref{Int64}()
    cutensornetNetworkOperatorAppendMPO(handle(), op, ComplexF64(coefficient), length(modes), convert(Vector{Int32}, modes),
                                        [collect(Int64, size(t)) for t in tensors], C_NULL, pointer.(tensors), boundary_condition, id)
    push!(op.tensors, tensors)
    return id[]
end

"""
    CuNetworkState{T}(extents; purity=CUTENSORNET_STATE_PURITY_PURE)

A quantum state, or other tensor network state, of modes with the given extents, starting as
the vacuum state (all modes 0). Apply operators to it with [`apply!`](@ref) and the other
`apply_*!` functions, or start from an MPS with [`initialize_mps!`](@ref). Then compute the
state itself with [`prepare!`](@ref) and [`compute!`](@ref), or properties of it with
[`CuStateAccessor`](@ref), [`CuStateExpectation`](@ref), [`CuStateMarginal`](@ref),
[`CuStateSampler`](@ref) and [`CuStateProjectionMPS`](@ref). With
`purity = CUTENSORNET_STATE_PURITY_MIXED`, it is a density matrix.

State modes are numbered from 0. The state keeps what it is given alive, but uses the
memory of operators directly: register changes to an operator with [`update!`](@ref).
That memory must be device memory.

Wraps `cutensornetState_t`.
"""
mutable struct CuNetworkState{T}
    handle::cutensornetState_t
    extents::Vector{Int64}
    purity::cutensornetStatePurity_t
    # data handed to the library, by the id of the operator it belongs to
    tensors::Dict{Int64,Any}
    gradients::Dict{Int64,Any}
    initial_tensors::Vector{Any}
    # the requested MPS factorization, if any
    mps_extents::Union{Nothing,Vector{Vector{Int64}}}
    output::Vector{CuArray{T}}
    workspace::Union{Nothing,Workspace}
    users::Int
    retired::Bool
    destroy::Function
    # operators that have been applied, and must outlive the state
    operators::Vector{CuNetworkOperator{T}}
    function CuNetworkState{T}(extents; purity::cutensornetStatePurity_t=CUTENSORNET_STATE_PURITY_PURE) where {T}
        state_ref = Ref{cutensornetState_t}()
        extents = collect(Int64, extents)
        cutensornetCreateState(handle(), purity, length(extents), extents, T, state_ref)
        destroy = function (state)
            cutensornetDestroyState(state)
            foreach(release!, state.operators)
        end
        obj = new{T}(state_ref[], extents, purity, Dict{Int64,Any}(), Dict{Int64,Any}(), Any[], nothing, CuArray{T}[], nothing, 0, false, destroy, CuNetworkOperator{T}[])
        resource_finalizer(retire!, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetState_t}, state::CuNetworkState) = state.handle
Base.eltype(::CuNetworkState{T}) where {T} = T

# the extents of the state tensor: those of the modes, followed by those of the dual modes
# for a mixed state (a density matrix)
state_extents(state::CuNetworkState) =
    state.purity == CUTENSORNET_STATE_PURITY_MIXED ? vcat(state.extents, state.extents) : state.extents

"""
    apply!(state::CuNetworkState{T}, modes, operator::CuArray{T};
           immutable=true, adjoint=false, unitary=false) -> Int64
    apply!(state::CuNetworkState{T}, operator::CuNetworkOperator{T}; ...) -> Int64

Apply an operator to the state, and return its id. A tensor `operator` acts on the 0-based
state `modes`, and has twice as many dimensions: its elements are stored following the rows
of its textbook definition, i.e. as `reshape(permutedims(M), ...)` of a matrix `M`.
`immutable = false` allows changing it later with [`update!`](@ref), `adjoint` applies its
adjoint instead, and `unitary` promises that it is unitary, which saves work.

Wraps `cutensornetStateApplyTensorOperator` and `cutensornetStateApplyNetworkOperator`.
"""
function apply!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operator::CuArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(modes) || throw(DimensionMismatch("an operator on $(length(modes)) modes must have $(2 * length(modes)) dimensions."))
    id = Ref{Int64}()
    cutensornetStateApplyTensorOperator(handle(), state, length(modes), convert(Vector{Int32}, modes), operator, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = operator
    return id[]
end

"""
    apply_with_gradient!(state::CuNetworkState{T}, modes, operator::CuArray{T},
                         gradient::CuArray{T}; immutable=true, adjoint=false, unitary=false)
        -> Int64

Apply an operator to the state as [`apply!`](@ref) does, and have
[`compute_with_gradients!`](@ref) on an expectation value of the state write the gradient
with respect to the operator to `gradient`.

Wraps `cutensornetStateApplyTensorOperatorWithGradient`.
"""
function apply_with_gradient!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operator::CuArray{T}, gradient::CuArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(modes) || throw(DimensionMismatch("an operator on $(length(modes)) modes must have $(2 * length(modes)) dimensions."))
    size(gradient) == size(operator) || throw(DimensionMismatch("the gradient must have the size of the operator."))
    id = Ref{Int64}()
    cutensornetStateApplyTensorOperatorWithGradient(handle(), state, length(modes), convert(Vector{Int32}, modes), operator, C_NULL, immutable, adjoint, unitary, gradient, C_NULL, id)
    state.tensors[id[]] = operator
    state.gradients[id[]] = gradient
    return id[]
end

"""
    update_gradient!(state::CuNetworkState{T}, id, gradient::CuArray{T}) -> state

Write the gradient with respect to the operator `id` to a different array.

Wraps `cutensornetStateUpdateTensorOperatorGradient`.
"""
function update_gradient!(state::CuNetworkState{T}, id::Integer, gradient::CuArray{T}) where {T}
    cutensornetStateUpdateTensorOperatorGradient(handle(), state, id, gradient)
    state.gradients[id] = gradient
    return state
end

"""
    apply_diagonal!(state::CuNetworkState{T}, modes, diagonal::CuArray{T};
                    immutable=true, adjoint=false, unitary=false) -> Int64

Apply a diagonal operator to the state, given by its diagonal as a tensor with one dimension
per mode, and return its id. Afterwards, [`compute!`](@ref) can't compute the full state
anymore, only properties of it.

Wraps `cutensornetStateApplyDiagonalTensorOperator`.
"""
function apply_diagonal!(state::CuNetworkState{T}, modes::Vector{<:Integer}, diagonal::CuArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(diagonal) == length(modes) || throw(DimensionMismatch("a diagonal operator on $(length(modes)) modes must have $(length(modes)) dimensions."))
    id = Ref{Int64}()
    cutensornetStateApplyDiagonalTensorOperator(handle(), state, length(modes), convert(Vector{Int32}, modes), diagonal, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = diagonal
    return id[]
end

"""
    apply_controlled!(state::CuNetworkState{T}, control_modes, target_modes, operator::CuArray{T};
                      control_values=nothing, immutable=true, adjoint=false, unitary=false)
        -> Int64

Apply `operator` to the `target_modes` of the state when the `control_modes` have the
`control_values`, which default to the last basis state of each mode (1 for qubits), and
return its id. `operator` only covers the target modes, laid out as for [`apply!`](@ref).

Wraps `cutensornetStateApplyControlledTensorOperator`.
"""
function apply_controlled!(state::CuNetworkState{T}, control_modes::Vector{<:Integer}, target_modes::Vector{<:Integer}, operator::CuArray{T}; control_values::Union{Nothing,Vector{<:Integer}}=nothing, immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(target_modes) || throw(DimensionMismatch("an operator on $(length(target_modes)) modes must have $(2 * length(target_modes)) dimensions."))
    values = control_values === nothing ? C_NULL : convert(Vector{Int64}, control_values)
    id = Ref{Int64}()
    cutensornetStateApplyControlledTensorOperator(handle(), state, length(control_modes), convert(Vector{Int32}, control_modes), values, length(target_modes), convert(Vector{Int32}, target_modes), operator, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = operator
    return id[]
end

"""
    update!(state::CuNetworkState{T}, id, operator::CuArray{T}; unitary=false) -> state

Use new data for the operator `id`, which must have been applied with `immutable = false`.
Also call this after changing the operator in place.

Wraps `cutensornetStateUpdateTensorOperator`.
"""
function update!(state::CuNetworkState{T}, id::Integer, operator::CuArray{T}; unitary::Bool=false) where {T}
    cutensornetStateUpdateTensorOperator(handle(), state, id, operator, unitary)
    state.tensors[id] = operator
    return state
end

function apply!(state::CuNetworkState{T}, operator::CuNetworkOperator{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    id = Ref{Int64}()
    cutensornetStateApplyNetworkOperator(handle(), state, operator, immutable, adjoint, unitary, id)
    state.tensors[id[]] = operator
    push!(state.operators, retain!(operator))
    return id[]
end

"""
    apply_unitary_channel!(state::CuNetworkState{T}, modes, operators::Vector{<:CuArray{T}},
                           probabilities) -> Int64

Apply a unitary channel to the state: each of `operators`, laid out as for
[`apply!`](@ref), with the corresponding probability. Returns its id.

Wraps `cutensornetStateApplyUnitaryChannel`.
"""
function apply_unitary_channel!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operators::Vector{<:CuArray{T}}, probabilities::Vector{<:Real}) where {T}
    length(operators) == length(probabilities) || throw(DimensionMismatch("need a probability for each operator."))
    id = Ref{Int64}()
    cutensornetStateApplyUnitaryChannel(handle(), state, length(modes), convert(Vector{Int32}, modes), length(operators), pointer.(operators), C_NULL, convert(Vector{Cdouble}, probabilities), id)
    state.tensors[id[]] = operators
    return id[]
end

"""
    apply_general_channel!(state::CuNetworkState{T}, modes, operators::Vector{<:CuArray{T}})
        -> Int64

Apply a general channel to the state, given by its Kraus operators, laid out as for
[`apply!`](@ref). Returns its id.

Wraps `cutensornetStateApplyGeneralChannel`.
"""
function apply_general_channel!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operators::Vector{<:CuArray{T}}) where {T}
    id = Ref{Int64}()
    cutensornetStateApplyGeneralChannel(handle(), state, length(modes), convert(Vector{Int32}, modes), length(operators), pointer.(operators), C_NULL, id)
    state.tensors[id[]] = operators
    return id[]
end

"""
    initialize_mps!(state::CuNetworkState{T}, tensors::Vector{<:CuArray{T}};
                    boundary_condition=CUTENSORNET_BOUNDARY_CONDITION_OPEN) -> state

Start the state from a matrix product state with one tensor per state mode, instead of from
the vacuum state. Each tensor has the modes (left bond, state mode, right bond), without the
outer bonds at the boundaries of an open chain.

Wraps `cutensornetStateInitializeMPS`.
"""
function initialize_mps!(state::CuNetworkState{T}, tensors::Vector{<:CuArray{T}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    length(tensors) == length(state.extents) || throw(DimensionMismatch("need one tensor for each state mode."))
    cutensornetStateInitializeMPS(handle(), state, boundary_condition, [collect(Int64, size(t)) for t in tensors], C_NULL, pointer.(tensors))
    state.initial_tensors = tensors
    return state
end

"""
    finalize_mps!(state::CuNetworkState, extents::Vector{<:Vector{<:Integer}};
                  boundary_condition=CUTENSORNET_BOUNDARY_CONDITION_OPEN,
                  svd_algorithm=CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ) -> state

Have [`compute!`](@ref) compute the state as a matrix product state, with tensors of at most
the given extents, laid out as for [`initialize_mps!`](@ref). The SVD used to truncate it
can be configured with [`configure!`](@ref) and `CUTENSORNET_STATE_CONFIG_MPS_SVD_*`; the
algorithm defaults to Jacobi (`gesvdj`), as NVIDIA's default fails for some states.

Wraps `cutensornetStateFinalizeMPS`.
"""
function finalize_mps!(state::CuNetworkState, extents::Vector{<:Vector{<:Integer}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN, svd_algorithm::cutensornetTensorSVDAlgo_t=CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ)
    length(extents) == length(state.extents) || throw(DimensionMismatch("need extents for each state mode."))
    state.mps_extents = [collect(Int64, e) for e in extents]
    cutensornetStateFinalizeMPS(handle(), state, boundary_condition, state.mps_extents, C_NULL)
    # the library's default SVD algorithm, gesvd, fails with an internal error when
    # computing the MPS, even with the host workspace it asks for attached.
    configure!(state, CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, svd_algorithm)
    return state
end

"""
    capture_mps!(state::CuNetworkState) -> state

Replace the state by the last MPS computed by [`compute!`](@ref), dropping the operators
applied so far.

Wraps `cutensornetStateCaptureMPS`.
"""
function capture_mps!(state::CuNetworkState)
    cutensornetStateCaptureMPS(handle(), state)
    empty!(state.tensors)
    empty!(state.gradients)
    return state
end

"""
    details(state::CuNetworkState) -> (; extents, strides)

The extents and strides of the tensors that [`compute!`](@ref) computes, one entry per
tensor.

Wraps `cutensornetGetOutputStateDetails`.
"""
function details(state::CuNetworkState)
    num_tensors = Ref{Int32}()
    cutensornetGetOutputStateDetails(handle(), state, num_tensors, C_NULL, C_NULL, C_NULL)
    num_modes = Vector{Int32}(undef, num_tensors[])
    cutensornetGetOutputStateDetails(handle(), state, num_tensors, num_modes, C_NULL, C_NULL)
    extents = [Vector{Int64}(undef, n) for n in num_modes]
    strides = [Vector{Int64}(undef, n) for n in num_modes]
    cutensornetGetOutputStateDetails(handle(), state, num_tensors, num_modes, extents, strides)
    return (; extents, strides)
end


## things that are computed from a state, with a `prepare!`, `configure!` and `info`

abstract type StateComputation end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, x::StateComputation) = x.handle

for (type, prefix, handle_type, attr_type, dtor) in (
        (:CuNetworkState,       :State,              :cutensornetState_t,              :cutensornetStateAttributes_t,              nothing),
        (:CuStateAccessor,      :Accessor,           :cutensornetStateAccessor_t,      :cutensornetAccessorAttributes_t,           :cutensornetDestroyAccessor),
        (:CuStateExpectation,   :Expectation,        :cutensornetStateExpectation_t,   :cutensornetExpectationAttributes_t,        :cutensornetDestroyExpectation),
        (:CuStateMarginal,      :Marginal,           :cutensornetStateMarginal_t,      :cutensornetMarginalAttributes_t,           :cutensornetDestroyMarginal),
        (:CuStateSampler,       :Sampler,            :cutensornetStateSampler_t,       :cutensornetSamplerAttributes_t,            :cutensornetDestroySampler),
        (:CuStateProjectionMPS, :StateProjectionMPS, :cutensornetStateProjectionMPS_t, :cutensornetStateProjectionMPSAttributes_t, :cutensornetDestroyStateProjectionMPS))
    configure = Symbol(:cutensornet, prefix, :Configure)
    prepare = Symbol(:cutensornet, prefix, :Prepare)
    get_info = Symbol(:cutensornet, prefix, :GetInfo)
    if dtor !== nothing
        @eval begin
            mutable struct $type{T} <: StateComputation
                handle::$handle_type
                # the library refers to the state, and any other data, without owning it
                state::Any
                data::Any
                owners::Vector{Any}
                workspace::Union{Nothing,Workspace}
                function $type{T}(handle, state, data, owners=[state]) where {T}
                    foreach(retain!, owners)
                    obj = new{T}(handle, state, data, owners, nothing)
                    resource_finalizer(obj) do x
                        $dtor(x)
                        foreach(release!, x.owners)
                    end
                    return obj
                end
            end
        end
    end
    @eval begin
        configure!(x::$type, attr::$attr_type, value) = (attr_buf = Ref(value); $configure(handle(), x, attr, attr_buf, sizeof(attr_buf)); x)
        function info(x::$type, attr::$attr_type, ::Type{R}) where {R}
            attr_buf = Ref{R}()
            $get_info(handle(), x, attr, attr_buf, sizeof(R))
            return attr_buf[]
        end
        function prepare!(x::$type; max_workspace::Integer=DEFAULT_MAX_WORKSPACE, preference::cutensornetWorksizePref_t=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED, stream::CuStream=stream())
            CUDACore.check_owned(stream)
            workspace = Workspace()
            $prepare(handle(), x, max_workspace, workspace, stream)
            x.workspace = allocate!(workspace; preference)
            return x
        end
    end
end

"""
    configure!(x, attr, value) -> x

Set a configuration attribute of a state or a computation on it. The attribute is given by
NVIDIA's enum for the type of `x` (`cutensornetStateAttributes_t`,
`cutensornetAccessorAttributes_t`, ...), e.g.
`CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES`, and `value` must have the type that
NVIDIA documents for it. Call it before [`prepare!`](@ref).

Wraps `cutensornetStateConfigure`, `cutensornetAccessorConfigure` and the like.
"""
configure!

"""
    info(x, attr, R) -> R

Read an information attribute of a prepared state or computation on it, such as
`CUTENSORNET_EXPECTATION_INFO_FLOPS`, as a value of type `R`.

Wraps `cutensornetStateGetInfo`, `cutensornetAccessorGetInfo` and the like.
"""
info

"""
    prepare!(x; max_workspace=2^30, preference=CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
             stream=stream()) -> x

Prepare a state, or a computation on it, for [`compute!`](@ref) (or [`sample!`](@ref),
[`compute_env!`](@ref) and the like), using at most `max_workspace` bytes of workspace,
which this allocates. Prepare a state again after applying more operators to it, or
changing whether one is `unitary`. Applying an operator to a state invalidates the
computations created on it before (accessors, expectations, marginals, samplers): create
those again.

Wraps `cutensornetStatePrepare`, `cutensornetAccessorPrepare` and the like.
"""
prepare!

"""
    flops(x) -> Float64

The estimated number of floating-point operations of the prepared state or computation.
"""
function flops end

flops(x::CuNetworkState)     = info(x, CUTENSORNET_STATE_INFO_FLOPS, Float64)
flops(x::CuStateAccessor)    = info(x, CUTENSORNET_ACCESSOR_INFO_FLOPS, Float64)
flops(x::CuStateExpectation) = info(x, CUTENSORNET_EXPECTATION_INFO_FLOPS, Float64)
flops(x::CuStateMarginal)    = info(x, CUTENSORNET_MARGINAL_INFO_FLOPS, Float64)
flops(x::CuStateSampler)     = info(x, CUTENSORNET_SAMPLER_INFO_FLOPS, Float64)
"""
    marginal_kind(x::CuStateMarginal) -> cutensornetMarginalKind_t

Whether the prepared marginal is computed as a full density matrix or as its diagonal.
"""
marginal_kind(x::CuStateMarginal) = info(x, CUTENSORNET_MARGINAL_INFO_KIND, cutensornetMarginalKind_t)

function require_prepared(x)
    x.workspace === nothing && throw(ArgumentError("call `prepare!` first."))
    return x.workspace
end

"""
    compute!(state::CuNetworkState; stream=stream()) -> (; tensors, extents, strides)
    compute!(acc::CuStateAccessor, projected_values=[]; stream=stream()) -> amplitudes, norm
    compute!(x::CuStateExpectation; stream=stream()) -> value, norm
    compute!(x::CuStateMarginal, projected_values=[]; stream=stream()) -> marginal

Compute a prepared state, or a property of it, and wait for the result.

- For a state, the result is the tensor of the full state, or the tensors of its MPS after
  [`finalize_mps!`](@ref). MPS tensors can be truncated: the returned extents and strides
  say which part of each tensor holds the data. The full state can't be computed after
  [`apply_diagonal!`](@ref). Wraps `cutensornetStateCompute`.
- For an accessor, the amplitudes of the modes that are not projected, for the projected
  modes having the given values, and the squared norm of the state. Wraps
  `cutensornetAccessorCompute`.
- For an expectation value, the expectation value of the operator, not divided by the
  squared norm of the state, which is also returned. Wraps `cutensornetExpectationCompute`.
- For a marginal, the reduced density matrix of its modes (or its diagonal), for the
  projected modes having the given values. Wraps `cutensornetMarginalCompute`.
"""
function compute!(state::CuNetworkState{T}; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    workspace = require_prepared(state)
    extents = state.mps_extents === nothing ? [copy(state_extents(state))] : deepcopy(state.mps_extents)
    strides = [zeros(Int64, length(e)) for e in extents]
    state.output = [CuArray{T}(undef, e...) for e in extents]
    cutensornetStateCompute(handle(), state, workspace, extents, strides, pointer.(state.output), stream)
    synchronize(stream)
    return (; tensors=state.output, extents, strides)
end

"""
    CuStateAccessor(state::CuNetworkState, projected_modes=Int32[])

A computation of amplitudes of the state, for the `projected_modes` having values that are
given to [`compute!`](@ref). The amplitudes have the remaining modes in order; for a mixed
state, these are the modes of the density matrix, with the dual modes numbered after the
others. Prepare it with [`prepare!`](@ref).

Wraps `cutensornetCreateAccessor`.
"""
function CuStateAccessor(state::CuNetworkState{T}, projected_modes::Vector{<:Integer}=Int32[]) where {T}
    acc_ref = Ref{cutensornetStateAccessor_t}()
    cutensornetCreateAccessor(handle(), state, length(projected_modes), convert(Vector{Int32}, projected_modes), C_NULL, acc_ref)
    # the amplitudes have the modes that aren't projected, in order. for mixed states, these
    # are the modes of the density matrix, with the dual modes numbered after the others.
    extents = [e for (m, e) in enumerate(state_extents(state)) if !((m - 1) in projected_modes)]
    return CuStateAccessor{T}(acc_ref[], state, (; extents, n_projected=length(projected_modes)))
end

function check_projection(x, values)
    length(values) == x.data.n_projected || throw(DimensionMismatch("need a value for each of the $(x.data.n_projected) projected modes."))
end

# the amplitudes for the projected modes having certain values, and the squared state norm
function compute!(acc::CuStateAccessor{T}, projected_values::Vector{<:Integer}=Int64[]; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    check_projection(acc, projected_values)
    amplitudes = CuArray{T}(undef, acc.data.extents...)
    norm = Ref{T}(zero(T))
    cutensornetAccessorCompute(handle(), acc, convert(Vector{Int64}, projected_values), require_prepared(acc), amplitudes, norm, stream)
    synchronize(stream)
    return amplitudes, real(norm[])
end

"""
    CuStateExpectation(state::CuNetworkState{T}, operator::CuNetworkOperator{T})

A computation of the expectation value of `operator` on the state, by [`compute!`](@ref)
or, with gradients, [`compute_with_gradients!`](@ref). Prepare it with [`prepare!`](@ref).

Wraps `cutensornetCreateExpectation`.
"""
function CuStateExpectation(state::CuNetworkState{T}, operator::CuNetworkOperator{T}) where {T}
    exp_ref = Ref{cutensornetStateExpectation_t}()
    cutensornetCreateExpectation(handle(), state, operator, exp_ref)
    return CuStateExpectation{T}(exp_ref[], state, operator, [state, operator])
end

function compute!(x::CuStateExpectation{T}; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    value = Ref{T}(zero(T))
    norm = Ref{T}(zero(T))
    cutensornetExpectationCompute(handle(), x, require_prepared(x), value, norm, stream)
    synchronize(stream)
    return value[], real(norm[])
end

"""
    compute_with_gradients!(x::CuStateExpectation; expectation_adjoint=1, norm_adjoint=0,
                            accumulate=false, stream=stream()) -> value, norm

Compute the expectation value as [`compute!`](@ref) does, and write the gradients with
respect to the operators applied with [`apply_with_gradient!`](@ref). The adjoints are the
derivatives of the end result of the computation with respect to the two return values.
With `accumulate`, the gradients are added to the arrays.

Wraps `cutensornetExpectationComputeWithGradientsBackward`.
"""
function compute_with_gradients!(x::CuStateExpectation{T}; expectation_adjoint::Number=one(T), norm_adjoint::Number=zero(T), accumulate::Bool=false, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    value = Ref{T}(zero(T))
    norm = Ref{T}(zero(T))
    adj_value = Ref{T}(expectation_adjoint)
    adj_norm = Ref{T}(norm_adjoint)
    cutensornetExpectationComputeWithGradientsBackward(handle(), x, accumulate, adj_value, adj_norm, require_prepared(x), value, norm, stream)
    synchronize(stream)
    return value[], real(norm[])
end

"""
    CuStateMarginal(state::CuNetworkState, modes; projected_modes=Int32[], diagonal=false)

A computation of the reduced density matrix of the state on `modes`. With `diagonal = true`,
only its diagonal is computed, i.e. the probabilities of the basis states. The
`projected_modes` take the values given to [`compute!`](@ref). Prepare it with
[`prepare!`](@ref).

Wraps `cutensornetCreateMarginal` and `cutensornetCreateMarginalDiagonal`.
"""
function CuStateMarginal(state::CuNetworkState{T}, modes::Vector{<:Integer}; projected_modes::Vector{<:Integer}=Int32[], diagonal::Bool=false) where {T}
    marginal_ref = Ref{cutensornetStateMarginal_t}()
    create = diagonal ? cutensornetCreateMarginalDiagonal : cutensornetCreateMarginal
    create(handle(), state, length(modes), convert(Vector{Int32}, modes), length(projected_modes), convert(Vector{Int32}, projected_modes), C_NULL, marginal_ref)
    extents = [state.extents[m + 1] for m in modes]
    return CuStateMarginal{T}(marginal_ref[], state, (; extents=diagonal ? extents : vcat(extents, extents), n_projected=length(projected_modes)))
end

function compute!(x::CuStateMarginal{T}, projected_values::Vector{<:Integer}=Int64[]; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    check_projection(x, projected_values)
    marginal = CuArray{T}(undef, x.data.extents...)
    cutensornetMarginalCompute(handle(), x, convert(Vector{Int64}, projected_values), require_prepared(x), marginal, stream)
    synchronize(stream)
    return marginal
end

"""
    CuStateSampler(state::CuNetworkState, modes)

A sampler of the values of `modes` in the state, for [`sample!`](@ref). Prepare it with
[`prepare!`](@ref).

Wraps `cutensornetCreateSampler`.
"""
function CuStateSampler(state::CuNetworkState{T}, modes::Vector{<:Integer}) where {T}
    sampler_ref = Ref{cutensornetStateSampler_t}()
    cutensornetCreateSampler(handle(), state, length(modes), convert(Vector{Int32}, modes), sampler_ref)
    return CuStateSampler{T}(sampler_ref[], state, length(modes))
end

"""
    sample!(x::CuStateSampler, num_shots; stream=stream()) -> Matrix{Int64}

Draw `num_shots` samples of the sampler's modes, as a matrix with a column of mode values for
each sample. Configure `CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC` with [`configure!`](@ref)
for reproducible samples.

Wraps `cutensornetSamplerSample`.
"""
function sample!(x::CuStateSampler, num_shots::Integer; stream::CuStream=stream())
    CUDACore.check_owned(stream)
    samples = Matrix{Int64}(undef, x.data, num_shots)
    cutensornetSamplerSample(handle(), x, num_shots, require_prepared(x), samples, stream)
    synchronize(stream)
    return samples
end


## MPS projection

env_bounds(env::Pair{<:Integer,<:Integer}) = cutensornetMPSEnvBounds_t(first(env), last(env))

"""
    CuStateProjectionMPS(state::CuNetworkState{T}, dual_tensors::Vector{<:CuArray{T}},
                         envs::Vector{<:Pair}; coefficient=1, symmetric=false, ortho=nothing,
                         boundary_condition=CUTENSORNET_BOUNDARY_CONDITION_OPEN)

A projection of the state on the MPS with the given dual tensors, for example to compress it
or to optimize the MPS site by site. Each environment is given as `lower => upper`, the
sites left and right of the MPS tensors it excludes, where `-1` and `length(dual_tensors)`
are the boundaries; it cannot span the whole MPS. `ortho` gives the orthogonality center of
the MPS, if it has one, in the same form: `0 => 2` for site 1. Prepare it with [`prepare!`](@ref), then use
[`tensor_info`](@ref), [`compute_env!`](@ref), [`extract_tensor!`](@ref) and
[`insert_tensor!`](@ref).

Wraps `cutensornetCreateStateProjectionMPS`.
"""
function CuStateProjectionMPS(state::CuNetworkState{T}, dual_tensors::Vector{<:CuArray{T}}, envs::Vector{<:Pair}; coefficient::Number=one(ComplexF64), symmetric::Bool=false, ortho::Union{Nothing,Pair}=nothing, boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    # the library segfaults on these when preparing the projection
    any(env -> first(env) == -1 && last(env) == length(dual_tensors), envs) && throw(ArgumentError("an environment cannot span the entire MPS."))
    projection_ref = Ref{cutensornetStateProjectionMPS_t}()
    ortho_spec = ortho === nothing ? C_NULL : Ref(env_bounds(ortho))
    cutensornetCreateStateProjectionMPS(handle(), 1, [state.handle], [ComplexF64(coefficient)], symmetric, length(envs), env_bounds.(envs), boundary_condition, length(dual_tensors), C_NULL,
                                        [collect(Int64, size(t)) for t in dual_tensors], C_NULL, pointer.(dual_tensors), ortho_spec, projection_ref)
    return CuStateProjectionMPS{T}(projection_ref[], state, dual_tensors)
end

"""
    tensor_info(x::CuStateProjectionMPS, env::Pair) -> (; extents, strides)

The extents of the MPS tensor excluded by the environment `env`, and the strides NVIDIA
recommends for it.

Wraps `cutensornetStateProjectionMPSGetTensorInfo`.
"""
function tensor_info(x::CuStateProjectionMPS, env::Pair)
    n_sites = last(env) - first(env) - 1
    at_boundary = first(env) == -1 || last(env) == length(x.data)
    n_modes = n_sites + 2 - (at_boundary ? 1 : 0)
    extents = Vector{Int64}(undef, n_modes)
    strides = Vector{Int64}(undef, n_modes)
    cutensornetStateProjectionMPSGetTensorInfo(handle(), x, Ref(env_bounds(env)), extents, strides)
    return (; extents, strides)
end

"""
    compute_env!(x::CuStateProjectionMPS{T}, env::Pair, input, output::CuArray{T};
                 apply_inv_metric=false, re_resolve_channels=false, stream=stream()) -> output

Compute the environment `env` of the projection into `output`, contracted with the MPS
tensor `input` if it isn't `nothing`.

Wraps `cutensornetStateProjectionMPSComputeTensorEnv`.
"""
function compute_env!(x::CuStateProjectionMPS{T}, env::Pair, input::Union{Nothing,CuArray{T}}, output::CuArray{T}; apply_inv_metric::Bool=false, re_resolve_channels::Bool=false, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    cutensornetStateProjectionMPSComputeTensorEnv(handle(), x, Ref(env_bounds(env)), C_NULL, something(input, CU_NULL), C_NULL, output, apply_inv_metric, re_resolve_channels, require_prepared(x), stream)
    synchronize(stream)
    return output
end

"""
    extract_tensor!(output::CuArray{T}, x::CuStateProjectionMPS{T}, env::Pair; stream=stream())
        -> output

Extract the MPS tensor excluded by the environment `env` into `output`.

Wraps `cutensornetStateProjectionMPSExtractTensor`.
"""
function extract_tensor!(output::CuArray{T}, x::CuStateProjectionMPS{T}, env::Pair; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    cutensornetStateProjectionMPSExtractTensor(handle(), x, Ref(env_bounds(env)), C_NULL, output, require_prepared(x), stream)
    synchronize(stream)
    return output
end

"""
    insert_tensor!(x::CuStateProjectionMPS{T}, env::Pair, input::CuArray{T}; ortho=nothing,
                   stream=stream()) -> x

Insert `input` as the MPS tensor excluded by the environment `env`, moving the
orthogonality center to `ortho`, given as for [`CuStateProjectionMPS`](@ref) (`0 => 2` for
site 1). It is the site of a one-site environment (the default), one of the sites of a
two-site environment, or a site next to the bond of a zero-site environment.

Wraps `cutensornetStateProjectionMPSInsertTensor`.
"""
function insert_tensor!(x::CuStateProjectionMPS{T}, env::Pair, input::CuArray{T}; ortho::Union{Nothing,Pair}=nothing, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    if ortho === nothing
        last(env) - first(env) == 2 || throw(ArgumentError("the orthogonality center must be specified for environments that don't exclude a single site."))
        ortho = env
    end
    cutensornetStateProjectionMPSInsertTensor(handle(), x, Ref(env_bounds(env)), Ref(env_bounds(ortho)), C_NULL, input, require_prepared(x), stream)
    synchronize(stream)
    return x
end

"""
    update_coefficient!(x::CuStateProjectionMPS, coefficient) -> x

Change the coefficient that the state is multiplied with in the projection.

Wraps `cutensornetStateProjectionMPSUpdateCoefficients`.
"""
function update_coefficient!(x::CuStateProjectionMPS, coefficient::Number)
    cutensornetStateProjectionMPSUpdateCoefficients(handle(), x, 1, [ComplexF64(coefficient)])
    return x
end

"""
    update_dual_tensors!(x::CuStateProjectionMPS{T}, dual_tensors::Vector{<:CuArray{T}};
                         ortho=nothing, stream=stream()) -> x

Replace the dual tensors of the projection by others with the same extents, with the
orthogonality center at `ortho` (given as for [`CuStateProjectionMPS`](@ref)), or none.

Wraps `cutensornetStateProjectionMPSUpdateDualTensors`.
"""
function update_dual_tensors!(x::CuStateProjectionMPS{T}, dual_tensors::Vector{<:CuArray{T}}; ortho::Union{Nothing,Pair}=nothing, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    extents = [collect(Int64, size(t)) for t in dual_tensors]
    ortho_spec = ortho === nothing ? Ref(env_bounds(-1 => length(dual_tensors))) : Ref(env_bounds(ortho))
    cutensornetStateProjectionMPSUpdateDualTensors(handle(), x, extents, extents, C_NULL, pointer.(dual_tensors), ortho_spec, stream)
    x.data = dual_tensors
    return x
end

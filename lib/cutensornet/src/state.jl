# the state API, for simulating quantum circuits and other tensor network states.
#
# state modes are 0-based, and tensor operators act on them following NVIDIA's conventions:
# with the default layout, the elements of an operator are stored in the row-following order
# of its textbook definition. all data handed to the library must stay alive for as long as
# the object it was given to, which these wrappers take care of.

const DEFAULT_MAX_WORKSPACE = 2^30

# for arrays of pointers to data that may live on the device or on the host, which the
# headers don't specify, and that can't be expressed by the type of the argument
rawptr(a::CuArray) = reinterpret(Ptr{Cvoid}, pointer(a))
rawptr(a::Array)   = Ptr{Cvoid}(pointer(a))

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

# a sum of terms, each a coefficient times a product of tensors acting on some state modes.
# returns the id of the term.
function append_product!(op::CuNetworkOperator{T}, coefficient::Number, modes::Vector{<:Vector{<:Integer}}, tensors::Vector{<:CuArray{T}}) where {T}
    length(modes) == length(tensors) || throw(DimensionMismatch("need as many tensors as sets of modes."))
    id = Ref{Int64}()
    cutensornetNetworkOperatorAppendProduct(handle(), op, ComplexF64(coefficient), length(tensors), Int32.(length.(modes)),
                                            [convert(Vector{Int32}, m) for m in modes], C_NULL, pointer.(tensors), id)
    push!(op.tensors, tensors)
    return id[]
end

function append_mpo!(op::CuNetworkOperator{T}, coefficient::Number, modes::Vector{<:Integer}, tensors::Vector{<:CuArray{T}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    length(modes) == length(tensors) || throw(DimensionMismatch("need one tensor for each mode."))
    id = Ref{Int64}()
    cutensornetNetworkOperatorAppendMPO(handle(), op, ComplexF64(coefficient), length(modes), convert(Vector{Int32}, modes),
                                        [collect(Int64, size(t)) for t in tensors], C_NULL, pointer.(tensors), boundary_condition, id)
    push!(op.tensors, tensors)
    return id[]
end

mutable struct CuNetworkState{T}
    handle::cutensornetState_t
    extents::Vector{Int64}
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
        obj = new{T}(state_ref[], extents, Dict{Int64,Any}(), Dict{Int64,Any}(), Any[], nothing, CuArray{T}[], nothing, 0, false, destroy, CuNetworkOperator{T}[])
        resource_finalizer(retire!, obj)
        return obj
    end
end
Base.unsafe_convert(::Type{cutensornetState_t}, state::CuNetworkState) = state.handle
Base.eltype(::CuNetworkState{T}) where {T} = T

function apply!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operator::AbstractArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(modes) || throw(DimensionMismatch("an operator on $(length(modes)) modes must have $(2 * length(modes)) dimensions."))
    id = Ref{Int64}()
    cutensornetStateApplyTensorOperator(handle(), state, length(modes), convert(Vector{Int32}, modes), operator, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = operator
    return id[]
end

# the gradient is written by `compute_with_gradients!` on an expectation value of the state
function apply_with_gradient!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operator::AbstractArray{T}, gradient::CuArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(modes) || throw(DimensionMismatch("an operator on $(length(modes)) modes must have $(2 * length(modes)) dimensions."))
    size(gradient) == size(operator) || throw(DimensionMismatch("the gradient must have the size of the operator."))
    id = Ref{Int64}()
    cutensornetStateApplyTensorOperatorWithGradient(handle(), state, length(modes), convert(Vector{Int32}, modes), operator, C_NULL, immutable, adjoint, unitary, gradient, C_NULL, id)
    state.tensors[id[]] = operator
    state.gradients[id[]] = gradient
    return id[]
end

function update_gradient!(state::CuNetworkState{T}, id::Integer, gradient::CuArray{T}) where {T}
    cutensornetStateUpdateTensorOperatorGradient(handle(), state, id, gradient)
    state.gradients[id] = gradient
    return state
end

# the operator is a tensor of the diagonal values, with a single dimension for each mode.
# computing the full state isn't supported afterwards, as noted at `compute!`.
function apply_diagonal!(state::CuNetworkState{T}, modes::Vector{<:Integer}, diagonal::AbstractArray{T}; immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(diagonal) == length(modes) || throw(DimensionMismatch("a diagonal operator on $(length(modes)) modes must have $(length(modes)) dimensions."))
    id = Ref{Int64}()
    cutensornetStateApplyDiagonalTensorOperator(handle(), state, length(modes), convert(Vector{Int32}, modes), diagonal, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = diagonal
    return id[]
end

# only the target tensor is provided. control values default to the last basis state.
function apply_controlled!(state::CuNetworkState{T}, control_modes::Vector{<:Integer}, target_modes::Vector{<:Integer}, operator::AbstractArray{T}; control_values::Union{Nothing,Vector{<:Integer}}=nothing, immutable::Bool=true, adjoint::Bool=false, unitary::Bool=false) where {T}
    ndims(operator) == 2 * length(target_modes) || throw(DimensionMismatch("an operator on $(length(target_modes)) modes must have $(2 * length(target_modes)) dimensions."))
    values = control_values === nothing ? C_NULL : convert(Vector{Int64}, control_values)
    id = Ref{Int64}()
    cutensornetStateApplyControlledTensorOperator(handle(), state, length(control_modes), convert(Vector{Int32}, control_modes), values, length(target_modes), convert(Vector{Int32}, target_modes), operator, C_NULL, immutable, adjoint, unitary, id)
    state.tensors[id[]] = operator
    return id[]
end

# register new data for an operator, also when it was changed in place
function update!(state::CuNetworkState{T}, id::Integer, operator::AbstractArray{T}; unitary::Bool=false) where {T}
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

function apply_unitary_channel!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operators::Vector{<:AbstractArray{T}}, probabilities::Vector{<:Real}) where {T}
    length(operators) == length(probabilities) || throw(DimensionMismatch("need a probability for each operator."))
    id = Ref{Int64}()
    cutensornetStateApplyUnitaryChannel(handle(), state, length(modes), convert(Vector{Int32}, modes), length(operators), rawptr.(operators), C_NULL, convert(Vector{Cdouble}, probabilities), id)
    state.tensors[id[]] = operators
    return id[]
end

function apply_general_channel!(state::CuNetworkState{T}, modes::Vector{<:Integer}, operators::Vector{<:AbstractArray{T}}) where {T}
    id = Ref{Int64}()
    cutensornetStateApplyGeneralChannel(handle(), state, length(modes), convert(Vector{Int32}, modes), length(operators), rawptr.(operators), C_NULL, id)
    state.tensors[id[]] = operators
    return id[]
end

# start from an MPS. each tensor has modes (left bond, state mode, right bond), without the
# outer bonds at the boundaries.
function initialize_mps!(state::CuNetworkState{T}, tensors::Vector{<:CuArray{T}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    length(tensors) == length(state.extents) || throw(DimensionMismatch("need one tensor for each state mode."))
    cutensornetStateInitializeMPS(handle(), state, boundary_condition, [collect(Int64, size(t)) for t in tensors], C_NULL, rawptr.(tensors))
    state.initial_tensors = tensors
    return state
end

# request `compute!` to factorize the state as an MPS with tensors of (at most) these extents.
#
# the library's default SVD algorithm, gesvd, fails with an internal error whenever it has
# to truncate or decompose: unlike with the tensor decompositions in tensornet.jl, the host
# workspace it needs since CUDA 13.4 Update 1 can't make it there. use gesvdj instead.
function finalize_mps!(state::CuNetworkState, extents::Vector{<:Vector{<:Integer}}; boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN, svd_algorithm::cutensornetTensorSVDAlgo_t=CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ)
    length(extents) == length(state.extents) || throw(DimensionMismatch("need extents for each state mode."))
    state.mps_extents = [collect(Int64, e) for e in extents]
    cutensornetStateFinalizeMPS(handle(), state, boundary_condition, state.mps_extents, C_NULL)
    configure!(state, CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO, svd_algorithm)
    return state
end

# drop all operators, and restart from the last MPS computed by `compute!`
function capture_mps!(state::CuNetworkState)
    cutensornetStateCaptureMPS(handle(), state)
    empty!(state.tensors)
    empty!(state.gradients)
    return state
end

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

flops(x::CuNetworkState)     = info(x, CUTENSORNET_STATE_INFO_FLOPS, Float64)
flops(x::CuStateAccessor)    = info(x, CUTENSORNET_ACCESSOR_INFO_FLOPS, Float64)
flops(x::CuStateExpectation) = info(x, CUTENSORNET_EXPECTATION_INFO_FLOPS, Float64)
flops(x::CuStateMarginal)    = info(x, CUTENSORNET_MARGINAL_INFO_FLOPS, Float64)
flops(x::CuStateSampler)     = info(x, CUTENSORNET_SAMPLER_INFO_FLOPS, Float64)
marginal_kind(x::CuStateMarginal) = info(x, CUTENSORNET_MARGINAL_INFO_KIND, cutensornetMarginalKind_t)

function require_prepared(x)
    x.workspace === nothing && throw(ArgumentError("call `prepare!` first."))
    return x.workspace
end

# compute the state, as a vector with a tensor for each MPS site, or the single tensor of the
# full state. the library doesn't support the latter for states with diagonal operators, for
# which the state's properties can be computed instead. MPS tensors can be truncated, as reported by the returned extents and strides,
# which also tell how the data is laid out in the tensors with their maximal extents.
function compute!(state::CuNetworkState{T}; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    workspace = require_prepared(state)
    extents = state.mps_extents === nothing ? [copy(state.extents)] : deepcopy(state.mps_extents)
    strides = [zeros(Int64, length(e)) for e in extents]
    state.output = [CuArray{T}(undef, e...) for e in extents]
    cutensornetStateCompute(handle(), state, workspace, extents, strides, pointer.(state.output), stream)
    synchronize(stream)
    return (; tensors=state.output, extents, strides)
end

function CuStateAccessor(state::CuNetworkState{T}, projected_modes::Vector{<:Integer}=Int32[]) where {T}
    acc_ref = Ref{cutensornetStateAccessor_t}()
    cutensornetCreateAccessor(handle(), state, length(projected_modes), convert(Vector{Int32}, projected_modes), C_NULL, acc_ref)
    # the amplitudes have the modes that aren't projected, in order
    extents = [e for (m, e) in enumerate(state.extents) if !((m - 1) in projected_modes)]
    return CuStateAccessor{T}(acc_ref[], state, extents)
end

# the amplitudes for the projected modes having certain values, and the squared state norm
function compute!(acc::CuStateAccessor{T}, projected_values::Vector{<:Integer}=Int64[]; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    amplitudes = CuArray{T}(undef, acc.data...)
    norm = Ref{T}(zero(T))
    cutensornetAccessorCompute(handle(), acc, convert(Vector{Int64}, projected_values), require_prepared(acc), amplitudes, norm, stream)
    synchronize(stream)
    return amplitudes, real(norm[])
end

function CuStateExpectation(state::CuNetworkState{T}, operator::CuNetworkOperator{T}) where {T}
    exp_ref = Ref{cutensornetStateExpectation_t}()
    cutensornetCreateExpectation(handle(), state, operator, exp_ref)
    return CuStateExpectation{T}(exp_ref[], state, operator, [state, operator])
end

# the unnormalized expectation value, and the squared state norm
function compute!(x::CuStateExpectation{T}; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    value = Ref{T}(zero(T))
    norm = Ref{T}(zero(T))
    cutensornetExpectationCompute(handle(), x, require_prepared(x), value, norm, stream)
    synchronize(stream)
    return value[], real(norm[])
end

# also write the gradients registered with `apply_with_gradient!`. the adjoints are the
# derivatives of the end result of a computation with respect to these two outputs.
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

function CuStateMarginal(state::CuNetworkState{T}, modes::Vector{<:Integer}; projected_modes::Vector{<:Integer}=Int32[], diagonal::Bool=false) where {T}
    marginal_ref = Ref{cutensornetStateMarginal_t}()
    create = diagonal ? cutensornetCreateMarginalDiagonal : cutensornetCreateMarginal
    create(handle(), state, length(modes), convert(Vector{Int32}, modes), length(projected_modes), convert(Vector{Int32}, projected_modes), C_NULL, marginal_ref)
    extents = [state.extents[m + 1] for m in modes]
    return CuStateMarginal{T}(marginal_ref[], state, diagonal ? extents : vcat(extents, extents))
end

function compute!(x::CuStateMarginal{T}, projected_values::Vector{<:Integer}=Int64[]; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    marginal = CuArray{T}(undef, x.data...)
    cutensornetMarginalCompute(handle(), x, convert(Vector{Int64}, projected_values), require_prepared(x), marginal, stream)
    synchronize(stream)
    return marginal
end

function CuStateSampler(state::CuNetworkState{T}, modes::Vector{<:Integer}) where {T}
    sampler_ref = Ref{cutensornetStateSampler_t}()
    cutensornetCreateSampler(handle(), state, length(modes), convert(Vector{Int32}, modes), sampler_ref)
    return CuStateSampler{T}(sampler_ref[], state, length(modes))
end

# returns a matrix with a column of mode values for each sample
function sample!(x::CuStateSampler, num_shots::Integer; stream::CuStream=stream())
    CUDACore.check_owned(stream)
    samples = Matrix{Int64}(undef, x.data, num_shots)
    cutensornetSamplerSample(handle(), x, num_shots, require_prepared(x), samples, stream)
    synchronize(stream)
    return samples
end


## MPS projection

env_bounds(env::Pair{<:Integer,<:Integer}) = cutensornetMPSEnvBounds_t(first(env), last(env))

# project a state on an MPS with the given dual tensors, for example to compress it. each
# environment is specified as `lower => upper`, the sites left and right of the excluded
# MPS tensors.
function CuStateProjectionMPS(state::CuNetworkState{T}, dual_tensors::Vector{<:CuArray{T}}, envs::Vector{<:Pair}; coefficient::Number=one(ComplexF64), symmetric::Bool=false, ortho::Union{Nothing,Pair}=nothing, boundary_condition::cutensornetBoundaryCondition_t=CUTENSORNET_BOUNDARY_CONDITION_OPEN) where {T}
    # the library segfaults on these when preparing the projection
    any(env -> first(env) == -1 && last(env) == length(dual_tensors), envs) && throw(ArgumentError("an environment cannot span the entire MPS."))
    projection_ref = Ref{cutensornetStateProjectionMPS_t}()
    ortho_spec = ortho === nothing ? C_NULL : Ref(env_bounds(ortho))
    cutensornetCreateStateProjectionMPS(handle(), 1, [state.handle], [ComplexF64(coefficient)], symmetric, length(envs), env_bounds.(envs), boundary_condition, length(dual_tensors), C_NULL,
                                        [collect(Int64, size(t)) for t in dual_tensors], C_NULL, pointer.(dual_tensors), ortho_spec, projection_ref)
    return CuStateProjectionMPS{T}(projection_ref[], state, dual_tensors)
end

# the extents of the MPS representation tensor of an environment, and the recommended strides
function tensor_info(x::CuStateProjectionMPS, env::Pair)
    n_sites = last(env) - first(env) - 1
    at_boundary = first(env) == -1 || last(env) == length(x.data)
    n_modes = n_sites + 2 - (at_boundary ? 1 : 0)
    extents = Vector{Int64}(undef, n_modes)
    strides = Vector{Int64}(undef, n_modes)
    cutensornetStateProjectionMPSGetTensorInfo(handle(), x, Ref(env_bounds(env)), extents, strides)
    return (; extents, strides)
end

function compute_env!(x::CuStateProjectionMPS{T}, env::Pair, input::Union{Nothing,CuArray{T}}, output::CuArray{T}; apply_inv_metric::Bool=false, re_resolve_channels::Bool=false, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    cutensornetStateProjectionMPSComputeTensorEnv(handle(), x, Ref(env_bounds(env)), C_NULL, something(input, CU_NULL), C_NULL, output, apply_inv_metric, re_resolve_channels, require_prepared(x), stream)
    synchronize(stream)
    return output
end

function extract_tensor!(output::CuArray{T}, x::CuStateProjectionMPS{T}, env::Pair; stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    cutensornetStateProjectionMPSExtractTensor(handle(), x, Ref(env_bounds(env)), C_NULL, output, require_prepared(x), stream)
    synchronize(stream)
    return output
end

function insert_tensor!(x::CuStateProjectionMPS{T}, env::Pair, input::CuArray{T}; ortho::Pair=env, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    cutensornetStateProjectionMPSInsertTensor(handle(), x, Ref(env_bounds(env)), Ref(env_bounds(ortho)), C_NULL, input, require_prepared(x), stream)
    synchronize(stream)
    return x
end

function update_coefficient!(x::CuStateProjectionMPS, coefficient::Number)
    cutensornetStateProjectionMPSUpdateCoefficients(handle(), x, 1, [ComplexF64(coefficient)])
    return x
end

# replace the dual tensors by others with the same extents
function update_dual_tensors!(x::CuStateProjectionMPS{T}, dual_tensors::Vector{<:CuArray{T}}; ortho::Union{Nothing,Pair}=nothing, stream::CuStream=stream()) where {T}
    CUDACore.check_owned(stream)
    extents = [collect(Int64, size(t)) for t in dual_tensors]
    ortho_spec = ortho === nothing ? Ref(env_bounds(-1 => length(dual_tensors))) : Ref(env_bounds(ortho))
    cutensornetStateProjectionMPSUpdateDualTensors(handle(), x, extents, extents, C_NULL, pointer.(dual_tensors), ortho_spec, stream)
    x.data = dual_tensors
    return x
end

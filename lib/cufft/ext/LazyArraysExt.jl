# Fusing broadcasts into transforms, spelled with LazyArrays' `@~`:
#
#     y = abs2.(@~ p * (w .* x))
#
# `@~ p * arg` builds a `LazyArrays.Applied` node, with `arg` an array or a lazy broadcast.
# The node has a broadcast style of its own, so that broadcasts that consume it end up
# here. The broadcast inside the node then becomes a cuFFT load callback, and the one around
# it a store callback, both linked into the transform (see `callbacks.jl`). When that
# isn't possible, the node is replaced by `p * materialize(arg)`, and the broadcast around
# it executes as usual.
module LazyArraysExt

using cuFFT
using cuFFT: CuFFTPlan, ScaledPlan, callback_issue, execute_fused!, fuse_broadcasts,
             input_type, output_type, unfused
using CUDACore
using CUDACore: Adapt, DenseCuArray, unsafe_free!
using LazyArrays: LazyArrays, Applied
using Base.Broadcast: Broadcast, Broadcasted, BroadcastStyle


## the lazy transform

const FusablePlan = Union{CuFFTPlan,ScaledPlan{<:Any,<:CuFFTPlan}}
const LazyFFT = Applied{<:Any,typeof(*),<:Tuple{FusablePlan,Any}}

# (the generic method takes microseconds to select a style, which doesn't matter here)
LazyArrays.applied(::typeof(*), p::FusablePlan, x) =
    Applied{LazyArrays.DefaultArrayApplyStyle}(*, (p, x))

unscaled(p::CuFFTPlan) = p
unscaled(p::ScaledPlan) = p.p

Base.size(a::LazyFFT) = unscaled(a.args[1]).output_size
Base.axes(a::LazyFFT) = map(Base.OneTo, size(a))
Base.ndims(a::LazyFFT) = length(size(a))
Base.eltype(a::LazyFFT) = output_type(a.args[1])
Broadcast.broadcastable(a::LazyFFT) = a
Broadcast.instantiate(a::LazyFFT) = a

struct FFTStyle <: BroadcastStyle end
Broadcast.BroadcastStyle(::Type{<:LazyFFT}) = FFTStyle()
Broadcast.BroadcastStyle(style::FFTStyle, ::BroadcastStyle) = style
Broadcast.BroadcastStyle(style::FFTStyle, ::Broadcast.Unknown) = style

Base.copy(a::LazyFFT) = copy(Broadcast.instantiate(Broadcast.broadcasted(identity, a)))

function Base.copy(bc::Broadcasted{FFTStyle})
    dest = fuse!(bc, Broadcast.combine_eltypes(bc.f, bc.args), nothing)
    dest isa String || return dest
    unfused(dest)
    return Broadcast.materialize(eager(bc))
end

function Base.copyto!(dest::AbstractArray, bc::Broadcasted{FFTStyle})
    reason = fuse!(bc, eltype(dest), dest)
    reason isa String || return dest
    unfused(reason)
    return Broadcast.materialize!(dest, eager(bc))
end

# the same broadcast, with the transforms in it executed
eager(bc::Broadcasted) = Broadcast.broadcasted(bc.f, map(eager, bc.args)...)
eager(a::LazyFFT) = a.args[1] * input(a.args[2])
eager(x) = x
input(arg::Broadcasted) = Broadcast.materialize(arg)
input(arg) = arg

# the transforms in a broadcast
nodes(bc::Broadcasted) = nodes(bc.args...)
nodes(a::LazyFFT) = (a,)
nodes(x) = ()
nodes() = ()
nodes(x, xs...) = (nodes(x)..., nodes(xs...)...)

# whether any array in a broadcast satisfies `f`
anyarray(f, bc::Broadcasted) = anyarray(f, bc.args...)
anyarray(f, x::AbstractArray) = f(x)
anyarray(f, x) = false
anyarray(f) = false
anyarray(f, x, xs...) = anyarray(f, x) || anyarray(f, xs...)


## expressions that callbacks evaluate

# A broadcast is turned into a tree of calls that is evaluated for one linear index of the
# transform's input or output. Unlike a `Broadcasted`, it has no style or axes.
struct Call{F,A<:Tuple}
    f::F
    args::A
end
Adapt.adapt_structure(to, c::Call) =
    Call(Adapt.adapt(to, c.f), map(arg -> Adapt.adapt(to, arg), c.args))

# a type that is called, which wouldn't be a bits value
struct Construct{T} end
@inline (::Construct{T})(args...) where {T} = T(args...)
callee(f) = f
callee(::Type{T}) where {T} = Construct{T}()

# the element that cuFFT stores
struct Value end

# an array with the size of the transform's input or output, indexed linearly
struct Linear{A}
    x::A
end
Adapt.adapt_structure(to, a::Linear) = Linear(Adapt.adapt(to, a.x))

tree(bc::Broadcasted, dims) = Call(callee(bc.f), map(arg -> tree(arg, dims), bc.args))
tree(::LazyFFT, dims) = Value()
tree(a::AbstractArray, dims) = size(a) == dims ? Linear(a) : Broadcast.extrude(a)
tree(x, dims) = x

@inline evaluate(c::Call, i, dims, v) = c.f(evaluate_args(c.args, i, dims, v)...)
@inline evaluate(::Value, i, dims, v) = v
@inline evaluate(a::Linear, i, dims, v) = @inbounds a.x[i]
@inline evaluate(a, i, dims, v) =
    @inbounds Broadcast._broadcast_getindex(a, cartesian(dims, i))
@inline evaluate_args(::Tuple{}, i, dims, v) = ()
@inline evaluate_args(args::Tuple, i, dims, v) =
    (evaluate(args[1], i, dims, v), evaluate_args(Base.tail(args), i, dims, v)...)

# The Cartesian index for a linear one, for arrays of another size. Callbacks can't throw,
# so this divides without checking for zero, and with 32 bits, as 64-bit division is slow
# on GPUs.
@inline cartesian(dims::Dims, i) = CartesianIndex(subscripts(dims, (i - 1) % UInt32))
@inline subscripts(::Tuple{}, r) = ()
@inline function subscripts(dims::Dims, r)
    d = dims[1] % UInt32
    q = Core.Intrinsics.udiv_int(r, d)
    return ((r - q * d) % Int + 1, subscripts(Base.tail(dims), q)...)
end

# the callbacks, see `callbacks.jl`
struct Load{F}
    tree::F
end
Adapt.adapt_structure(to, l::Load) = Load(Adapt.adapt(to, l.tree))
@inline (l::Load)(x, i) = evaluate(l.tree, i, size(x), nothing)

struct Store{F,A}
    tree::F
    dest::A
end
Adapt.adapt_structure(to, s::Store) = Store(Adapt.adapt(to, s.tree), Adapt.adapt(to, s.dest))
@inline function (s::Store)(y, i, v)
    @inbounds s.dest[i] = evaluate(s.tree, i, size(y), v)
    return
end

# the broadcast around a transform, applied to an element of its output. a `ScaledPlan`
# normalizes that element first.
store_tree(bc, p::CuFFTPlan, dims) = tree(bc, dims)
store_tree(bc, p::ScaledPlan, dims) =
    scale(tree(bc, dims), convert(real(output_type(p)), p.scale))
scale(c::Call, s) = Call(c.f, map(arg -> scale(arg, s), c.args))
scale(::Value, s) = Call(*, (Value(), s))
scale(x, s) = x


## fusion

"""
    fuse!(bc, D, dest) -> Union{AbstractArray,String}

Execute the broadcast `bc`, which contains a lazy transform, with the broadcast fused into
the transform. The result is stored in `dest`, or in a new array with element type `D` if
`dest` is `nothing`, and returned. If the broadcast cannot be fused, nothing is executed
and the reason is returned.
"""
function fuse!(bc::Broadcasted{FFTStyle}, ::Type{D}, dest) where {D}
    fuse_broadcasts[] || return "fusion is disabled"
    found = nodes(bc)
    found isa Tuple{LazyFFT} || return "the broadcast contains more than one transform"
    return fuse!(bc, D, dest, found[1])
end

function fuse!(bc::Broadcasted{FFTStyle}, ::Type{D}, dest, node::LazyFFT) where {D}
    p = unscaled(node.args[1])
    issue = callback_issue(p)
    issue === nothing || return issue
    T, S = output_type(p), input_type(p)

    isbitstype(D) || return "the result is not a bits type"
    dest === nothing || dest isa DenseCuArray || return "the destination is not a dense CuArray"
    map(length, axes(bc)) == p.output_size ||
        return "the broadcast has another shape than the transform's output"
    max(prod(p.input_size), prod(p.output_size)) <= typemax(UInt32) ||
        return "the transform has too many elements"

    arg = node.args[2]
    arg isa Broadcasted && (arg = Broadcast.instantiate(arg))
    arg isa Union{Broadcasted,AbstractArray} || return "the transform's input is not an array"
    map(length, axes(arg)) == p.input_size ||
        return "the transform's input has another size than the plan's"

    # callbacks read their arrays while others write the destination
    if dest !== nothing
        anyarray(x -> Base.mightalias(dest, x), arg) &&
            return "the destination aliases the transform's input"
        anyarray(x -> x !== dest && Base.mightalias(dest, x), bc) &&
            return "the destination aliases another array"
    end

    # the input: an array cuFFT reads, or a load callback
    load = nothing
    staged = nothing
    if T <: Real
        # complex-to-real transforms overwrite their input, even when loading it with a
        # callback, so they get a copy
        x = staged = CuArray{S}(undef, p.input_size)
        x .= arg
    elseif arg isa DenseCuArray{S}
        x = arg
    else
        x = nothing
        load = Load(tree(arg, p.input_size))
    end

    # the output: the destination, or a store callback that writes it
    expr = store_tree(bc, node.args[1], p.output_size)
    direct = expr isa Call{typeof(identity),Tuple{Value}} && D === T
    if direct && load === nothing
        staged === nothing || unsafe_free!(staged)
        return "there is nothing to fuse"
    end
    result = dest === nothing ? CuArray{D}(undef, p.output_size) : dest
    store = direct ? nothing : Store(expr, result)
    # cuFFT may use its output array as scratch space before it calls the store callback,
    # so that array has to be as large as the transform's output, and the callback can't
    # read from it
    reads = anyarray(x -> x === dest, bc)
    y = D === T && !reads ? result : CuArray{T}(undef, p.output_size)

    issue = GC.@preserve arg bc execute_fused!(p, x, y; load, store)

    staged === nothing || unsafe_free!(staged)
    y === result || unsafe_free!(y)
    if issue !== nothing
        dest === nothing && unsafe_free!(result)
        return issue
    end
    return result
end

end

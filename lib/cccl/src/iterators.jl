# custom iterators
#
# Besides pointers, CCCL's algorithms take iterators defined by a block of state bytes and
# two device functions, which HostJIT inlines like operators. Both work through pointers:
#
#     void advance(void *state, const void *offset);
#     void dereference(const void *state, T *result);   // input iterators
#     void dereference(void *state, const void *value); // output iterators
#
# CUB copies the state into its kernels, so it must not refer to host memory.

# an iterator over values of type `T`; `state` needs to be kept alive while it is used
function custom_iterator(advance::Operator, dereference::Operator, ::Type{T},
                         state::Base.RefValue) where {T}
    S = typeof(state[])
    cccl_iterator_t(sizeof(S), Base.datatype_alignment(S), CCCL_ITERATOR, cccl_op(advance),
                    cccl_op(dereference), type_info(T),
                    Base.unsafe_convert(Ptr{Cvoid}, state), C_NULL)
end

# the functions of iterators without a position, which need no state; HostJIT copies some
# anyway
struct Stay end
@inline (::Stay)(state::OpPtr, offset::OpPtr) = nothing

struct LoadZero{T} end
@inline function (::LoadZero{T})(state::OpPtr, result::OpPtr) where {T}
    store!(result, zero(T))
    return
end

struct Discard end
@inline (::Discard)(state::OpPtr, value::OpPtr) = nothing

const stateless_state = Ref{UInt8}(0)

iterator_function(W) = cached_operator(W(), W, 2)

# an input iterator yielding zeros of type `T`
zeros_iterator(::Type{T}) where {T} =
    custom_iterator(iterator_function(Stay), iterator_function(LoadZero{T}), T,
                    stateless_state)

# an output iterator discarding the values of type `T` written to it
discard_iterator(::Type{T}) where {T} =
    custom_iterator(iterator_function(Stay), iterator_function(Discard), T, stateless_state)

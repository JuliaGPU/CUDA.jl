# finalizers are run out-of-order disregarding field references between objects (see
# JuliaLang/julia#3067), so we manually need to keep instances alive outside of the object
# fields in order to prevent objects getting collected before their owners are

# For example, we have:
#   type CuContext
#     ...
#   end
#   finalize(ctx::CuContext) = ccall(:cuDestroyContext, ctx)
#
#   type CuArray
#     ctx::CuContext
#     ptr::DevicePtr
#   end
#   finalize(arr::CuArray) = ccall(:cuMemFree, arr.ptr)
#
#  As finalizers are not guaranteed to run in order, the context might be finalized before
#  the array, but doing so breaks the call to :cuMemFree as destroying the underlying
#  context invalidates all resources.

# NOTE: the Dict is inversed (owner => target), to make the more common block/unblock cheap
# NOTE: we use `pointer_from_objref` instead of `WeakRef` (either of both are required
#       not to keep the owner artificially alive) because different objects can have
#       identical hashes
const finalizer_blocks = Dict{Ptr{Void},Any}()

# NOTE: in principle, `unblock_finalizer` doesn't need to know the target, but we force it
#       to keep the owner's object layout sane (ie. it will still need to have a field with
#       the target objects), which keeps the dependency explicit

function block_finalizer(owner::ANY, target::ANY...)
    ref = Base.pointer_from_objref(owner)
    trace("Blocking finalization of $(typeof(target)) at $((Base.pointer_from_objref(target))) by $(typeof(owner)) at $ref")
    haskey(finalizer_blocks, ref) && error("can only issue a single call to block_finalizer")
    finalizer_blocks[ref] = target
end

function unblock_finalizer(owner::ANY, target::ANY...)
    ref = Base.pointer_from_objref(owner)
    trace("Unblocking finalization of $(typeof(target)) at $((Base.pointer_from_objref(target))) by $(typeof(owner)) at $ref")
    haskey(finalizer_blocks, ref) || error("no finalizer blocks found (no call to block_finalizer, or has been unblocked already)")
    finalizer_blocks[ref] == target || error("mismatch between block and unblock_finalizer targets")
    delete!(finalizer_blocks, ref)
end

can_finalize(target::ANY) = isempty(filter((k,v) -> v == target, finalizer_blocks))

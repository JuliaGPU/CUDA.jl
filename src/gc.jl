# auxiliary infrastructure to cope with out-of-order finalization (JuliaLang/julia#3067)

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
#   As finalizers are not guaranteed to run in order, the context might be finalized before
#   the array, but doing so breaks the call to :cuMemFree as destroying the underlying
#   context invalidates all resources.
#
# The functions in this file help in construction additional dependency chains, blocking
# finalizers to be run before all child objects have been collected. Note that during
# process exit, even those additional dependency chains are violated (see `can_finalize`).

# NOTE: we use `pointer_from_objref` instead of `WeakRef` (either of both are required
#       not to keep the owner artificially alive) because different objects can have
#       identical hashes
const finalizer_blocks = Dict{Ptr{Void},Any}() # owner => target to make block/unblock cheap

# NOTE: I removed the support for registering multiple targets, because the splat was too
#       expensive. If we ever need this again, use dispatch to differentiate between
#       single/multiple target

"""
Block finalization of `target`, and register `owner` as the owning object.

This function is meant to be called in the constructor of a child object, which needs to be
finalized _before_ any parent object is finalized.
"""
function block_finalizer(owner::ANY, target::ANY)
    owner_id = Base.pointer_from_objref(owner)
    @trace("Blocking finalization of $target at $((Base.pointer_from_objref(target)))",
           " by $(typeof(owner)) at $owner_id")
    haskey(finalizer_blocks, owner_id) && error("can only issue a single call to block_finalizer")
    finalizer_blocks[owner_id] = target
end

"""
Unblock finalization of `target`, checking whether it was owned by `owner`.

This function is meant to be called in the finalized of a child object.
"""
function unblock_finalizer(owner::ANY, target::ANY)
    owner_id = Base.pointer_from_objref(owner)
    @trace("Unblocking finalization of $target at $((Base.pointer_from_objref(target)))",
           " by $(typeof(owner)) at $owner_id")
    haskey(finalizer_blocks, owner_id) ||
        error("no finalizer blocks found (no call to block_finalizer, or has been unblocked already)")

    # NOTE: in principle, `unblock_finalizer` doesn't need to know the target, but we force it
    #       to keep the owner's object layout sane (ie. it will still need to have a field with
    #       the target objects), which keeps the dependency explicit
    finalizer_blocks[owner_id] == target ||
        error("mismatch between block and unblock_finalizer targets")

    delete!(finalizer_blocks, owner_id)
end

"""
Check whether an object can be finalized, ie. whether it has not been blocked by any other object.

This function is meant to be called in the finalizer of the parent object, as during process
exit Julia's garbage collector ignores the dependency chains set-up by `block_finalizer` and
`unblock_finalizer`.
"""
can_finalize(target::ANY) = isempty(filter((owner,candidate) -> candidate == target, finalizer_blocks))

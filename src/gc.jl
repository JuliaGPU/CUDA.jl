# finalizers are run out-of-order disregarding field references between objects (see
# JuliaLang/julia#3067), so we manually need to keep instances alive outside of the object
# fields in order to prevent parent objects getting collected before their children

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
#  the array, but doing breaks the call to :cuMemFree as destroying the underlying context
#  invalidates all resources.

# NOTE: the Dict is inversed (child => parent), to make the more ccommon track/untrack cheap
# NOTE: we use `pointer_from_objref` instead of `WeakRef` because
#       different objects can have the same hash
const gc_keepalive = Dict{Ptr{Void},Any}()

function gc_track(parent::ANY, child::ANY)
    ref = Base.pointer_from_objref(child)
    trace("Tracking $(typeof(parent)) at $((Base.pointer_from_objref(parent))) from $(typeof(child)) at $ref")
    haskey(gc_keepalive, ref) && error("objects can only keep a single parent alive")
    gc_keepalive[ref] = parent
end

function gc_untrack(parent::ANY, child::ANY)
    ref = Base.pointer_from_objref(child)
    trace("Untracking $(typeof(parent)) at $((Base.pointer_from_objref(parent))) from $(typeof(child)) at $ref")
    haskey(gc_keepalive, ref) || error("track/untrack mismatch")
    gc_keepalive[ref] == parent || error("track/untrack mismatch on parent")
    delete!(gc_keepalive, ref)
end

gc_children(parent::ANY) = filter((k,v) -> v == parent, gc_keepalive)

@testset "gc" begin

# clear exception_in_transit, which can hold a reference to objects
# (this works around JuliaLang/julia#20784)
try throw(nothing) end

# test outstanding contexts
@test length(CUDAdrv.context_instances) == 1
destroy(ctx)
for i in 1:50
    gc()
end
if length(CUDAdrv.context_instances) > 0
    for (handle, object) in CUDAdrv.context_instances
        warn("CUDA context $handle has outstanding object at $object")
    end
end
@test length(CUDAdrv.context_instances) == 0

# test blocked finalizers
if length(CUDAdrv.finalizer_blocks) > 0
    for (owner_ptr, target) in CUDAdrv.finalizer_blocks
        # NOTE: we don't deref the owner ptr here, because it might have been freed
        #       (in the case that object forgot to unblock target during finalization)
        warn("Blocked finalizer: object at $owner_ptr keeps $target alive")
    end
end
@test length(CUDAdrv.finalizer_blocks) == 0

end

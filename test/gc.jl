@testset "gc" begin

# clear exception_in_transit, which can hold a reference to objects
# (this works around JuliaLang/julia#20784)
try throw(nothing) end

# test outstanding contexts
@test length(CUDAdrv.context_instances) == 1
destroy!(ctx)
for i in 1:50
    gc()
end
if length(CUDAdrv.context_instances) > 0
    for (handle, object) in CUDAdrv.context_instances
        warn("CUDA context $handle has outstanding object at $object")
    end
end
@test length(CUDAdrv.context_instances) == 0

end

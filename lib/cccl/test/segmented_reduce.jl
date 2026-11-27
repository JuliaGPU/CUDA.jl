x = rand(Float32, 1000)
offsets = [0, 3, 3, 10, 1000]
out = CuArray{Float32}(undef, length(offsets) - 1)
CCCL.segmented_reduce!(+, out, CuArray(x), CuArray(offsets); init=0f0)
@test Array(out) ≈ [sum(x[offsets[i]+1:offsets[i+1]]; init=0f0) for i in 1:length(offsets)-1]

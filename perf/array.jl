group = addgroup!(SUITE, "array")

group["construct"] = @benchmarkable CuArray{Int}(undef, 1)

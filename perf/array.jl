group = addgroup!(SUITE, "base")

group["construct"] = @benchmarkable CuArray{Int}(undef, 1)

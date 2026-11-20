x = rand(Int32(1):Int32(100), 10_000)
y = CuArray{Int32}(undef, length(x))
CCCL.inclusive_scan!(+, y, CuArray(x))
@test Array(y) == cumsum(x)
CCCL.inclusive_scan!(+, y, CuArray(x); init=Int32(10))
@test Array(y) == cumsum(x) .+ 10
CCCL.exclusive_scan!(+, y, CuArray(x); init=Int32(0))
@test Array(y) == [0; cumsum(x)[1:end-1]]
CCCL.inclusive_scan!(max, y, CuArray(x))
@test Array(y) == accumulate(max, x)

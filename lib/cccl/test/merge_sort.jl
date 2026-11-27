keys = rand(Float64, 10_000)
d_keys = CuArray(keys)
CCCL.merge_sort!(d_keys)
@test Array(d_keys) == sort(keys)

# comparators capturing state
let c = 0.5
    d_keys = CuArray(keys)
    CCCL.merge_sort!(d_keys; lt=(a, b) -> abs(a - c) < abs(b - c))
    @test Array(d_keys) == sort(keys; by=k -> abs(k - c))
end

d_keys = CuArray(keys)
d_values = CuArray(collect(1:length(keys)))
CCCL.merge_sort!(d_keys; lt=(>), values=d_values)
@test Array(d_keys) == sort(keys; rev=true)
@test keys[Array(d_values)] == Array(d_keys)

for T in (UInt8, Int32, UInt64, Float32, Float64)
    keys = rand(T, 10_000)
    d_keys = CuArray(keys)
    CCCL.radix_sort!(d_keys)
    @test Array(d_keys) == sort(keys)

    d_keys = CuArray(keys)
    d_values = CuArray(collect(Int32(1):Int32(length(keys))))
    CCCL.radix_sort!(d_keys; values=d_values, rev=true)
    @test Array(d_keys) == sort(keys; rev=true)
    @test keys[Array(d_values)] == Array(d_keys)
end

# sorting empty arrays first must not break later sorts of the same types
let keys = rand(Int16, 1000), values = rand(Int16, 1000)
    CCCL.radix_sort!(CuArray(Int16[]); values=CuArray(Int16[]))
    d_keys, d_values = CuArray(keys), CuArray(values)
    CCCL.radix_sort!(d_keys; values=d_values)
    @test Array(d_values) == values[sortperm(keys)]
end

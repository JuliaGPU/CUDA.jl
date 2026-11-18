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

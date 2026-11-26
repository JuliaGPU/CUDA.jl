using BFloat16s: BFloat16

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

# keys CUB has no matching type for, or orders differently than `isless`
sample(T, n) = T == Char ? rand('a':'z', n) :
               T <: AbstractFloat ? T.(randn(Float32, n)) : rand(T, n)
for T in (Bool, Char, Int8, UInt16, Int64, Float16, BFloat16, Float32, Float64)
    for n in (0, 1, 1000, 100_000)
        keys = sample(T, n)
        for rev in (false, true)
            d_keys = CuArray(keys)
            CCCL.radix_sort!(d_keys; rev)
            @test isequal(Array(d_keys), sort(keys; rev))

            d_keys = CuArray(keys)
            d_values = CuArray(collect(1:n))
            CCCL.radix_sort!(d_keys; values=d_values, rev)
            @test isequal(Array(d_keys), sort(keys; rev))
            @test Array(d_values) == sortperm(keys; rev)
        end
    end
end

# floating-point special values, including NaNs of both signs and with payloads
@testset "$T" for T in (Float16, BFloat16, Float32, Float64)
    U = Base.uinttype(T)
    nan_payload = reinterpret(T, reinterpret(U, T(NaN)) | one(U))
    specials = T[1, NaN, -0.0, 0.0, -Inf, -NaN, 2, Inf, -1, -nan_payload, nan_payload,
                 floatmin(T), -floatmin(T), nextfloat(zero(T)), -nextfloat(zero(T))]
    for keys in (specials, rand(specials, 10_000), [specials; T.(randn(Float32, 10_000))])
        for rev in (false, true)
            d_keys = CuArray(keys)
            CCCL.radix_sort!(d_keys; rev)
            sorted = Array(d_keys)
            @test isequal(sorted, sort(keys; rev))
            # the same keys, bit for bit
            @test sort(reinterpret(U, sorted)) == sort(reinterpret(U, keys))

            d_keys = CuArray(keys)
            d_values = CuArray(collect(1:length(keys)))
            CCCL.radix_sort!(d_keys; values=d_values, rev)
            @test isequal(Array(d_keys), sort(keys; rev))
            @test reinterpret(U, keys[Array(d_values)]) == reinterpret(U, Array(d_keys))
        end
    end
end

@test_throws ArgumentError CCCL.radix_sort!(CuArray(rand(Int128, 10)))

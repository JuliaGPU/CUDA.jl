using cuBLAS
using LinearAlgebra

@testset "deprecated cuBLASXt device set" begin
    devs = @test_deprecated cuBLAS.devices()
    @test devs == cuBLAS.xt_devices()
    @test (@test_deprecated cuBLAS.ndevices()) == cuBLAS.xt_ndevices()
    @test_deprecated cuBLAS.devices!(devs)
    @test cuBLAS.xt_devices() == devs
end

@testset "deprecated pointer methods of the batched LAPACK routines" begin
    n, batch = 4, 3
    A = [rand(Float32, n, n) + n*I for _ in 1:batch]
    B = [rand(Float32, n, 2) for _ in 1:batch]

    d_A = CuMatrix.(A)
    pivot, info = @test_deprecated cuBLAS.getrf_batched!(n, cuBLAS.unsafe_batch(d_A), n, true, nothing)
    @test all(iszero, collect(info))

    d_B = CuMatrix.(B)
    info = @test_deprecated cuBLAS.getrs_batched!('N', n, 2, cuBLAS.unsafe_batch(d_A), n,
                                                  pointer(pivot), cuBLAS.unsafe_batch(d_B), n)
    @test info[] == 0
    @test collect.(d_B) ≈ A .\ B

    d_C = [similar(a) for a in d_A]
    info = @test_deprecated cuBLAS.getri_batched!(n, cuBLAS.unsafe_batch(d_A), n,
                                                  cuBLAS.unsafe_batch(d_C), n, pivot)
    @test all(iszero, collect(info))
    @test collect.(d_C) ≈ inv.(A)
end

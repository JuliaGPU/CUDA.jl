using Random

n = 256

function apply_seed(seed)
    if seed === missing
        # should result in different numbers across launches
        Random.seed!()
        # XXX: this currently doesn't work, because of the definition in Base,
        #      `seed!(r::MersenneTwister=default_rng())`, which breaks overriding
        #      `default_rng` with a non-MersenneTwister RNG.
    elseif seed !== nothing
        # should result in the same numbers
        Random.seed!(seed)
    elseif seed === nothing
        # should result in different numbers across launches,
        # as determined by the seed set during module loading.
    end
end

@testset "rand($T), seed $seed" for T in (Int32, UInt32, Int64, UInt64, Int128, UInt128,
                                          Float16, Float32, Float64),
                                    seed in (nothing, #=missing,=# 1234)
    # different kernel invocations should get different numbers
    @testset "across launches" begin
        function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = threadIdx().x
            A[tid] = rand(T)
            return nothing
        end

        a = CUDA.zeros(T, n)
        b = CUDA.zeros(T, n)

        @cuda threads=n kernel(a, seed)
        @cuda threads=n kernel(b, seed)

        if seed === nothing || seed === missing
            @test Array(a) != Array(b)
        else
            @test Array(a) == Array(b)
        end
    end

    # multiple calls to rand should get different numbers
    @testset "across calls" begin
        function kernel(A::AbstractArray{T}, B::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            tid = threadIdx().x
            A[tid] = rand(T)
            B[tid] = rand(T)
            return nothing
        end

        a = CUDA.zeros(T, n)
        b = CUDA.zeros(T, n)

        @cuda threads=n kernel(a, b, seed)

        @test Array(a) != Array(b)
    end

    # different threads should get different numbers
    @testset "across threads" for active_dim in 1:6
        function kernel(A::AbstractArray{T}, seed) where {T}
            apply_seed(seed)
            id = threadIdx().x*threadIdx().y*threadIdx().z*blockIdx().x*blockIdx().y*blockIdx().z
            A[id] = rand(T)
            return nothing
        end

        tx, ty, tz, bx, by, bz = [dim == active_dim ? 2 : 1 for dim in 1:6]
        a = CUDA.zeros(T, 2)

        @cuda threads=(tx, ty, tz) blocks=(bx, by, bz) kernel(a, seed)

        @test Array(a)[1] != Array(a)[2]
    end
end

@testset "basic randn($T), seed $seed" for T in (Float16, Float32, Float64),
                                           seed in (nothing, #=missing,=# 1234)
    function kernel(A::AbstractArray{T}, seed) where {T}
        apply_seed(seed)
        tid = threadIdx().x
        A[tid] = randn(T)
        return
    end

    a = CUDA.zeros(T, n)
    b = CUDA.zeros(T, n)

    @cuda threads=n kernel(a, seed)
    @cuda threads=n kernel(b, seed)

    if seed === nothing || seed === missing
        @test Array(a) != Array(b)
    else
        @test Array(a) == Array(b)
    end
end

@testset "basic randexp($T), seed $seed" for T in (Float16, Float32, Float64),
                                           seed in (nothing, #=missing,=# 1234)
    function kernel(A::AbstractArray{T}, seed) where {T}
        apply_seed(seed)
        tid = threadIdx().x
        A[tid] = randexp(T)
        return
    end

    a = CUDA.zeros(T, n)
    b = CUDA.zeros(T, n)

    @cuda threads=n kernel(a, seed)
    @cuda threads=n kernel(b, seed)

    if seed === nothing || seed === missing
        @test Array(a) != Array(b)
    else
        @test Array(a) == Array(b)
    end
end



# a composite type to test for more complex element types
@eval struct RGB{T}
    r::T
    g::T
    b::T
end

n = 256

@testset "constructors" begin
    # static
    @on_device @cuStaticSharedMem(Float32, 1)
    @on_device @cuStaticSharedMem(Float32, (1,2))
    @on_device @cuStaticSharedMem(Tuple{Float32, Float32}, 1)
    @on_device @cuStaticSharedMem(Tuple{Float32, Float32}, (1,2))
    @on_device @cuStaticSharedMem(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device @cuStaticSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2))

    # dynamic
    @on_device @cuDynamicSharedMem(Float32, 1)
    @on_device @cuDynamicSharedMem(Float32, (1, 2))
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, 1)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, (1,2))
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, 1)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2))

    # dynamic with offset
    @on_device @cuDynamicSharedMem(Float32, 1, 8)
    @on_device @cuDynamicSharedMem(Float32, (1,2), 8)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, 1, 8)
    @on_device @cuDynamicSharedMem(Tuple{Float32, Float32}, (1,2), 8)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, 1, 8)
    @on_device @cuDynamicSharedMem(Tuple{RGB{Float32}, UInt32}, (1,2), 8)
end


@testset "dynamic shmem" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuDynamicSharedMem(Float32, n)
        s[t] = d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda threads=n shmem=n*sizeof(Float32) kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    @testset for T in [Int32, Int64, Float16, Float32, Float64]
        function kernel(d::CuDeviceArray{T}, n) where {T}
            t = threadIdx().x
            tr = n-t+1

            s = @cuDynamicSharedMem(T, n)
            s[t] = d[t]
            sync_threads()
            d[t] = s[tr]

            return
        end

        a = rand(T, n)
        d_a = CuArray(a)

        @cuda threads=n shmem=n*sizeof(T) kernel(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    function kernel(v0::T, n) where {T}
        shared = @cuDynamicSharedMem(T, n)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    n = 32
    typ = typeof((0f0, 0f0, 0f0))
    @cuda shmem=n*sizeof(typ) kernel((0f0, 0f0, 0f0), n)
end

end


@testset "static shmem" begin

@testset "statically typed" begin
    function kernel(d, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuStaticSharedMem(Float32, 1024)
        s2 = @cuStaticSharedMem(Float32, 1024)  # catch aliasing

        s[t] = d[t]
        s2[t] = 2*d[t]
        sync_threads()
        d[t] = s[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuArray(a)

    @cuda threads=n kernel(d_a, n)
    @test reverse(a) == Array(d_a)
end

@testset "parametrically typed" begin
    @testset for typ in [Int32, Int64, Float16, Float32, Float64]
        function kernel(d::CuDeviceArray{T}, n) where {T}
            t = threadIdx().x
            tr = n-t+1

            s = @cuStaticSharedMem(T, 1024)
            s2 = @cuStaticSharedMem(T, 1024)  # catch aliasing

            s[t] = d[t]
            s2[t] = d[t]
            sync_threads()
            d[t] = s[tr]

            return
        end

        a = rand(typ, n)
        d_a = CuArray(a)

        @cuda threads=n kernel(d_a, n)
        @test reverse(a) == Array(d_a)
    end
end

@testset "alignment" begin
    # bug: used to generate align=12, which is invalid (non pow2)
    function kernel(v0::T) where {T}
        shared = CUDA.@cuStaticSharedMem(T, 32)
        @inbounds shared[Cuint(1)] = v0
        return
    end

    @cuda kernel((0f0, 0f0, 0f0))
end

end


@testset "dynamic shmem consisting of multiple arrays" begin

# common use case 1: dynamic shmem consists of multiple homogeneous arrays
#                    -> split using `view`
@testset "homogeneous" begin
    function kernel(a, b, n)
        t = threadIdx().x
        tr = n-t+1

        s = @cuDynamicSharedMem(eltype(a), 2*n)

        sa = view(s, 1:n)
        sa[t] = a[t]
        sync_threads()
        a[t] = sa[tr]

        sb = view(s, n+1:2*n)
        sb[t] = b[t]
        sync_threads()
        b[t] = sb[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Float32, n)
    d_b = CuArray(b)

    @cuda threads=n shmem=2*n*sizeof(Float32) kernel(d_a, d_b, n)
    @test reverse(a) == Array(d_a)
    @test reverse(b) == Array(d_b)
end

# common use case 2: dynamic shmem consists of multiple heterogeneous arrays
#                    -> construct using pointer offset
@testset "heterogeneous" begin
    function kernel(a, b, n)
        t = threadIdx().x
        tr = n-t+1

        sa = @cuDynamicSharedMem(eltype(a), n)
        sa[t] = a[t]
        sync_threads()
        a[t] = sa[tr]

        sb = @cuDynamicSharedMem(eltype(b), n, n*sizeof(eltype(a)))
        sb[t] = b[t]
        sync_threads()
        b[t] = sb[tr]

        return
    end

    a = rand(Float32, n)
    d_a = CuArray(a)

    b = rand(Int64, n)
    d_b = CuArray(b)

    @cuda threads=n shmem=(n*sizeof(Float32) + n*sizeof(Int64)) kernel(d_a, d_b, n)
    @test reverse(a) == Array(d_a)
    @test reverse(b) == Array(d_b)
end

end

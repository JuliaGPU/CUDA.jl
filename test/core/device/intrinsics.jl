@testset "indexing" begin
    @on_device threadIdx().x
    @on_device blockDim().x
    @on_device blockIdx().x
    @on_device gridDim().x

    @on_device threadIdx().y
    @on_device blockDim().y
    @on_device blockIdx().y
    @on_device gridDim().y

    @on_device threadIdx().z
    @on_device blockDim().z
    @on_device blockIdx().z
    @on_device gridDim().z

    @on_device warpsize()
    @on_device laneid()
    @on_device active_mask()

    @testset "range metadata" begin
        foobar() = threadIdx().x
        ir = sprint(io->CUDA.code_llvm(io, foobar, Tuple{}; raw=true))

        @test occursin(r"call .+ @llvm.nvvm.read.ptx.sreg.tid.x.+ !range", ir)
    end
end


############################################################################################

@testset "assertion" begin
    function kernel(i)
        @cuassert i > 0
        @cuassert i > 0 "test"
        return
    end

    @cuda kernel(1)
end


############################################################################################

@testset "data movement and conversion" begin

if capability(device()) >= v"3.0"

@testset "shuffle idx" begin
    function kernel(d)
        i = threadIdx().x
        j = 32 - i + 1

        d[i] = shfl_sync(FULL_MASK, d[i], j)

        return
    end

    warpsize = CUDA.warpsize(device())

    @testset for T in [UInt8, UInt16, UInt32, UInt64, UInt128,
                       Int8, Int16, Int32, Int64, Int128,
                       Float16, Float32, Float64,
                       ComplexF32, ComplexF64, Bool]
        a = rand(T, warpsize)
        d_a = CuArray(a)
        @cuda threads=warpsize kernel(d_a)
        @test Array(d_a) == reverse(a)
    end
end

@testset "shuffle down" begin
    n = 14

    function kernel(d::CuDeviceArray, n)
        t = threadIdx().x
        if t <= n
            d[t] += shfl_down_sync(FULL_MASK, d[t], n÷2, 32)
        end
        return
    end

    @testset for T in [Int32, Float32]
        a = T[T(i) for i in 1:n]
        d_a = CuArray(a)

        threads = nextwarp(device(), n)
        @cuda threads kernel(d_a, n)

        a[1:n÷2] += a[n÷2+1:end]
        @test a == Array(d_a)
    end
end

end

end



############################################################################################

@testset "clock and nanosleep" begin

@on_device clock(UInt32)
@on_device clock(UInt64)

if capability(device()) >= v"7.0"
@on_device nanosleep(UInt32(16))
end

end

@testset "parallel synchronization and communication" begin

@on_device sync_threads()
@on_device sync_threads_count(Int32(1))
@on_device sync_threads_count(true)
@on_device sync_threads_and(Int32(1))
@on_device sync_threads_and(true)
@on_device sync_threads_or(Int32(1))
@on_device sync_threads_or(true)

@testset "llvm ir barrier int" begin
    function kernel_barrier_count(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_count(an_array_of_1[i]) == 3
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    b_in = Int32[1, 1, 1, 0, 0]  # 3 true
    d_b = CuArray(b_in)
    @cuda threads=5 kernel_barrier_count(d_b)
    b_out = Array(d_b)

    @test b_out == b_in .+ 1

    function kernel_barrier_and(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_and(an_array_of_1[i]) > 0
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    a_in = Int32[1, 1, 1, 1, 1]  # all true
    d_a = CuArray(a_in)
    @cuda threads=5 kernel_barrier_and(d_a)
    a_out = Array(d_a)

    @test a_out == a_in .+ 1

    function kernel_barrier_or(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_or(an_array_of_1[i]) > 0
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    c_in = Int32[1, 0, 0, 0, 0]  # 1 true
    d_c = CuArray(c_in)
    @cuda threads=5 kernel_barrier_or(d_c)
    c_out = Array(d_c)

    @test c_out == c_in .+ 1
end

@testset "llvm ir barrier bool" begin
    function kernel_barrier_count(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_count(an_array_of_1[i] > 0) == 3
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    b_in = Int32[1, 1, 1, 0, 0]  # 3 true
    d_b = CuArray(b_in)
    @cuda threads=5 kernel_barrier_count(d_b)
    b_out = Array(d_b)

    @test b_out == b_in .+ 1

    function kernel_barrier_and(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_and(an_array_of_1[i] > 0)
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    a_in = Int32[1, 1, 1, 1, 1]  # all true
    d_a = CuArray(a_in)
    @cuda threads=5 kernel_barrier_and(d_a)
    a_out = Array(d_a)

    @test a_out == a_in .+ 1

    function kernel_barrier_or(an_array_of_1)
        i = (blockIdx().x-1i32) * blockDim().x + threadIdx().x
        if sync_threads_or(an_array_of_1[i] > 0)
            an_array_of_1[i] += Int32(1)
        end
        return nothing
    end

    c_in = Int32[1, 0, 0, 0, 0]  # 1 true
    d_c = CuArray(c_in)
    @cuda threads=5 kernel_barrier_or(d_c)
    c_out = Array(d_c)

    @test c_out == c_in .+ 1
end

@on_device sync_warp()
@on_device sync_warp(0xffffffff)
@on_device threadfence_block()
@on_device threadfence()
@on_device threadfence_system()

@testset "voting" begin

@testset "any" begin
    d_a = CuArray([false])

    function kernel(a, i)
        vote = vote_any_sync(FULL_MASK, threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a)[]

    @cuda threads=2 kernel(d_a, 2)
    @test Array(d_a)[]

    @cuda threads=2 kernel(d_a, 3)
    @test !Array(d_a)[]
end

@testset "all" begin
    d_a = CuArray([false])

    function kernel(a, i)
        vote = vote_all_sync(FULL_MASK, threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a)[]

    @cuda threads=2 kernel(d_a, 2)
    @test !Array(d_a)[]

    @cuda threads=2 kernel(d_a, 3)
    @test !Array(d_a)[]
end

@testset "uni" begin
    d_a = CuArray([false])

    function kernel1(a, i)
        vote = vote_uni_sync(FULL_MASK, threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel1(d_a, 1)
    @test Array(d_a)[]

    @cuda threads=2 kernel1(d_a, 2)
    @test !Array(d_a)[]

    function kernel2(a, i)
        vote = vote_uni_sync(FULL_MASK, blockDim().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel2(d_a, 3)
    @test Array(d_a)[]
end

@testset "ballot" begin
    d_a = CuArray(UInt32[0])

    function kernel(a, i)
        vote = vote_ballot_sync(FULL_MASK, threadIdx().x == i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    len = 4
    for i in 1:len
        @cuda threads=len kernel(d_a, i)
        @test Array(d_a) == [2^(i-1)]
    end
end

end

end

############################################################################################

@testset "libcudadevrt" begin
    kernel() = (CUDA.device_synchronize(); nothing)
    @cuda kernel()
end

############################################################################################


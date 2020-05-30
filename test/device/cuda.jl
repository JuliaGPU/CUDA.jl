@testset "CUDA functionality" begin

############################################################################################

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

    @testset "range metadata" begin
        foobar() = threadIdx().x
        ir = sprint(io->CUDA.code_llvm(io, foobar, Tuple{}; raw=true))

        @test occursin(r"call .+ @llvm.nvvm.read.ptx.sreg.tid.x.+ !range", ir)
    end
end



############################################################################################

@testset "math" begin
    buf = CuArray(zeros(Float32))

    function kernel(a, i)
        a[] = CUDA.log10(i)
        return
    end

    @cuda kernel(buf, Float32(100))
    val = Array(buf)
    @test val[] ≈ 2.0


    @testset "pow" begin
        buf = CuArray(zeros(Float32))

        function pow_kernel(a, x, y)
            a[] = CUDA.pow(x, y)
            return
        end

        #pow(::Float32, ::Float32)
        x, y = rand(Float32), rand(Float32)
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y
        @cuda pow_kernel(buf, x, -y)
        val = Array(buf)
        @test val[] ≈ x^(-y)

        #pow(::Float64, ::Float64)
        x, y = rand(Float64), rand(Float64)
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y
        @cuda pow_kernel(buf, x, -y)
        val = Array(buf)
        @test val[] ≈ x^(-y)

        #pow(::Float32, ::Int32)
        x, y = rand(Float32), rand(Int32(5):Int32(10))
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y
        @cuda pow_kernel(buf, x, -y)
        val = Array(buf)
        @test val[] ≈ x^(-y)

        #pow(::Float64, ::Int32)
        x, y = rand(Float64), rand(Int32(5):Int32(10))
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y
        @cuda pow_kernel(buf, x, -y)
        val = Array(buf)
        @test val[] ≈ x^(-y)

        #pow(::Float32, ::Int64)
        x, y = rand(Float32), rand(5:10)
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y

        #pow(::Float64, ::Int64)
        x, y = rand(Float32), rand(5:10)
        @cuda pow_kernel(buf, x, y)
        val = Array(buf)
        @test val[] ≈ x^y
    end

    @testset "isinf" begin
      buf = CuArray(zeros(Bool))

      function isinf_kernel(a, x)
          a[] = CUDA.isinf(x)
          return
      end

      for x in (Inf32, Inf, NaN32, NaN)
        @cuda isinf_kernel(buf, x)
        val = Array(buf)
        @test val[] == isinf(x)
      end

      codeinfo_str = sprint(show, code_typed(CUDA.isinf,
                                             Tuple{Float32}, optimize=false)[1])
      @test !occursin("Float64", codeinfo_str)

      codeinfo_str = sprint(show, code_typed(CUDA.isinf,
                                             Tuple{Float64}, optimize=false)[1])
      @test !occursin("Float32", codeinfo_str)
    end

    @testset "isnan" begin
      buf = CuArray(zeros(Bool))

      function isnan_kernel(a, x)
          a[] = CUDA.isnan(x)
          return
      end

      for x in (Inf32, Inf, NaN32, NaN)
        @cuda isnan_kernel(buf, x)
        val = Array(buf)
        @test val[] == isnan(x)
      end

      codeinfo_str = sprint(show, code_typed(CUDA.isnan,
                                             Tuple{Float32}, optimize=false)[1])
      @test !occursin("Float64", codeinfo_str)

      codeinfo_str = sprint(show, code_typed(CUDA.isnan,
                                             Tuple{Float64}, optimize=false)[1])
      @test !occursin("Float32", codeinfo_str)
    end

    @testset "angle" begin
        buf  = CuArray(zeros(Float32))
        cbuf = CuArray(zeros(Float32))

        function cuda_kernel(a, x)
            a[] = CUDA.angle(x)
            return
        end

        #op(::Float32)
        x   = rand(Float32)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ angle(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ angle(-x)

        #op(::ComplexF32)
        x   = rand(ComplexF32)
        @cuda cuda_kernel(cbuf, x)
        val = Array(cbuf)
        @test val[] ≈ angle(x)
        @cuda cuda_kernel(cbuf, -x)
        val = Array(cbuf)
        @test val[] ≈ angle(-x)

        #op(::Float64)
        x   = rand(Float64)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ angle(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ angle(-x)

        #op(::ComplexF64)
        x   = rand(ComplexF64)
        @cuda cuda_kernel(cbuf, x)
        val = Array(cbuf)
        @test val[] ≈ angle(x)
        @cuda cuda_kernel(cbuf, -x)
        val = Array(cbuf)
        @test val[] ≈ angle(-x)
    end

    # dictionary of key=>tuple, where the tuple should
    # contain the cpu command and the cuda function to test.
    ops = Dict("exp"=>(exp, CUDA.exp),
               "angle"=>(angle, CUDA.angle),
               "exp2"=>(exp2, CUDA.exp2),
               "exp10"=>(exp10, CUDA.exp10),
               "expm1"=>(expm1, CUDA.expm1))

    @testset "$key" for key=keys(ops)
        cpu_op, cuda_op = ops[key]

        buf = CuArray(zeros(Float32))

        function cuda_kernel(a, x)
            a[] = cuda_op(x)
            return
        end

        #op(::Float32)
        x   = rand(Float32)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)

        #op(::Float64)
        x   = rand(Float64)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)
    end

    # dictionary of key=>tuple, where the tuple should
    # contain the cpu command and the cuda function to test.
    ops = Dict("exp"=>(exp, CUDA.exp),
               "abs"=>(abs, CUDA.abs),
               "abs2"=>(abs2, CUDA.abs2),
               "angle"=>(angle, CUDA.angle),
               "log"=>(log, CUDA.log))

    @testset "Complex - $key" for key=keys(ops)
        cpu_op, cuda_op = ops[key]

        buf = CuArray(zeros(Complex{Float32}))

        function cuda_kernel(a, x)
            a[] = cuda_op(x)
            return
        end

        #op(::ComplexF32, ::ComplexF32)
        x   = rand(ComplexF32)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)

        #op(::ComplexF64, ::ComplexF64)
        x   = rand(ComplexF64)
        @cuda cuda_kernel(buf, x)
        val = Array(buf)
        @test val[] ≈ cpu_op(x)
        @cuda cuda_kernel(buf, -x)
        val = Array(buf)
        @test val[] ≈ cpu_op(-x)
    end
end


############################################################################################

endline = Sys.iswindows() ? "\r\n" : "\n"

@testset "formatted output" begin
    _, out = @grab_output @on_device @cuprintf("")
    @test out == ""

    _, out = @grab_output @on_device @cuprintf("Testing...\n")
    @test out == "Testing...$endline"

    # narrow integer
    _, out = @grab_output @on_device @cuprintf("Testing %d %d...\n", Int32(1), Int32(2))
    @test out == "Testing 1 2...$endline"

    # wide integer
    _, out = @grab_output if Sys.iswindows()
        @on_device @cuprintf("Testing %lld %lld...\n", Int64(1), Int64(2))
    else
        @on_device @cuprintf("Testing %ld %ld...\n", Int64(1), Int64(2))
    end
    @test out == "Testing 1 2...$endline"

    _, out = @grab_output @on_device begin
        @cuprintf("foo")
        @cuprintf("bar\n")
    end
    @test out == "foobar$endline"

    # c argument promotions
    function kernel(A)
        @cuprintf("%f %f\n", A[1], A[1])
        return
    end
    x = CuArray(ones(2, 2))
    _, out = @grab_output begin
        @cuda kernel(x)
        synchronize()
    end
    @test out == "1.000000 1.000000$endline"
end

@testset "@cuprint" begin
    # basic @cuprint/@cuprintln

    _, out = @grab_output @on_device @cuprint("Hello, World\n")
    @test out == "Hello, World$endline"

    _, out = @grab_output @on_device @cuprintln("Hello, World")
    @test out == "Hello, World$endline"


    # argument interpolation (by the macro, so can use literals)

    _, out = @grab_output @on_device @cuprint("foobar")
    @test out == "foobar"

    _, out = @grab_output @on_device @cuprint(:foobar)
    @test out == "foobar"

    _, out = @grab_output @on_device @cuprint("foo", "bar")
    @test out == "foobar"

    _, out = @grab_output @on_device @cuprint("foobar ", 42)
    @test out == "foobar 42"

    _, out = @grab_output @on_device @cuprint("foobar $(42)")
    @test out == "foobar 42"

    _, out = @grab_output @on_device @cuprint("foobar $(4)", 2)
    @test out == "foobar 42"

    _, out = @grab_output @on_device @cuprint("foobar ", 4, "$(2)")
    @test out == "foobar 42"

    _, out = @grab_output @on_device @cuprint(42)
    @test out == "42"

    _, out = @grab_output @on_device @cuprint(4, 2)
    @test out == "42"

    # bug: @cuprintln failed to invokce @cuprint with endline in the case of interpolation
    _, out = @grab_output @on_device @cuprintln("foobar $(42)")
    @test out == "foobar 42$endline"


    # argument types

    # we're testing the generated functions now, so can't use literals
    function test_output(val, str)
        canary = rand(Int32) # if we mess up the main arg, this one will print wrong
        _, out = @grab_output @on_device @cuprint(val, " (", canary, ")")
        @test out == "$(str) ($(Int(canary)))"
    end

    for typ in (Int16, Int32, Int64, UInt16, UInt32, UInt64)
        test_output(typ(42), "42")
    end

    for typ in (Float32, Float64)
        test_output(typ(42), "42.000000")
    end

    test_output(Cchar('c'), "c")

    for typ in (Ptr{Cvoid}, Ptr{Int})
        ptr = convert(typ, Int(0x12345))
        test_output(ptr, Sys.iswindows() ? "0000000000012345" : "0x12345")
    end

    test_output(true, "1")
    test_output(false, "0")


    # escaping

    kernel1(val) = (@cuprint(val); nothing)
    _, out = @grab_output @on_device kernel1(42)
    @test out == "42"

    kernel2(val) = (@cuprintln(val); nothing)
    _, out = @grab_output @on_device kernel2(42)
    @test out == "42$endline"
end

@testset "@cushow" begin
    function use_cushow()
        seven_i32 = Int32(7)
        three_f64 = Float64(3)
        @cushow seven_i32
        @cushow three_f64
        @cushow 1f0 + 4f0
        return nothing
    end

    _, out = @grab_output @on_device use_cushow()
    @test out == "seven_i32 = 7$(endline)three_f64 = 3.000000$(endline)1.0f0 + 4.0f0 = 5.000000$(endline)"
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

# a composite type to test for more complex element types
@eval struct RGB{T}
    r::T
    g::T
    b::T
end

@testset "shared memory" begin

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
    @testset for T in [Int32, Int64, Float32, Float64]
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
    @testset for typ in [Int32, Int64, Float32, Float64]
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
                       Float32, Float64, ComplexF32, ComplexF64, Bool]
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
        @cuda threads=threads kernel(d_a, n)

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

if CUDA.release() >= v"10.0" && v"6.2" in CUDA.ptx_support()
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
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

@testset "ballot" begin
    d_a = CuArray(UInt32[0])

    function kernel(a, i)
        vote = vote_ballot(threadIdx().x == i)
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

@testset "any" begin
    d_a = CuArray(UInt32[0])

    function kernel(a, i)
        vote = vote_any(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 2)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 3)
    @test Array(d_a) == [0]
end

@testset "all" begin
    d_a = CuArray(UInt32[0])

    function kernel(a, i)
        vote = vote_all(threadIdx().x >= i)
        if threadIdx().x == 1
            a[1] = vote
        end
        return
    end

    @cuda threads=2 kernel(d_a, 1)
    @test Array(d_a) == [1]

    @cuda threads=2 kernel(d_a, 2)
    @test Array(d_a) == [0]

    @cuda threads=2 kernel(d_a, 3)
    @test Array(d_a) == [0]
end

end

end

############################################################################################

@testset "libcudadevrt" begin
    kernel() = (CUDA.device_synchronize(); nothing)
    @cuda kernel()
end

############################################################################################

@testset "atomics (low-level)" begin

@testset "atomic_add" begin
    types = [Int32, Int64, UInt32, UInt64, Float32]
    capability(device()) >= v"6.0" && push!(types, Float64)

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(a, b)
            CUDA.atomic_add!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == T(1024)
    end
end

@testset "atomic_sub" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"6.0" && append!(types, [Float32, Float64])

    @testset for T in types
        a = CuArray([T(2048)])

        function kernel(a, b)
            CUDA.atomic_sub!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == T(1024)
    end
end

@testset "atomic_inc" begin
    @testset for T in [Int32]
        a = CuArray([zero(T)])

        function kernel(a, b)
            CUDA.atomic_inc!(pointer(a), b)
            return
        end

        @cuda threads=768 kernel(a, T(512))
        @test Array(a)[1] == T(255)
    end
end

@testset "atomic_dec" begin
    @testset for T in [Int32]
        a = CuArray([T(1024)])

        function kernel(a, b)
            CUDA.atomic_dec!(pointer(a), b)
            return
        end

        @cuda threads=256 kernel(a, T(512))
        @test Array(a)[1] == T(257)
    end
end

@testset "atomic_xchg" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T)])

        function kernel(a, b)
            CUDA.atomic_xchg!(pointer(a), b)
            return
        end

        @cuda threads=1024 kernel(a, one(T))
        @test Array(a)[1] == one(T)
    end
end

@testset "atomic_and" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([T(1023)])

        function kernel(a, T)
            i = threadIdx().x - 1
            k = 1
            for i = 1:i
                k *= 2
            end
            b = 1023 - k  # 1023 - 2^i
            CUDA.atomic_and!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == zero(T)
    end
end

@testset "atomic_or" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T)])

        function kernel(a, T)
            i = threadIdx().x
            b = 1  # 2^(i-1)
            for i = 1:i
                b *= 2
            end
            b /= 2
            CUDA.atomic_or!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == T(1023)
    end
end

@testset "atomic_xor" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([T(1023)])

        function kernel(a, T)
            i = threadIdx().x
            b = 1  # 2^(i-1)
            for i = 1:i
                b *= 2
            end
            b /= 2
            CUDA.atomic_xor!(pointer(a), T(b))
            return
        end

        @cuda threads=10 kernel(a, T)
        @test Array(a)[1] == zero(T)
    end
end

if capability(device()) >= v"6.0"

@testset "atomic_cas" begin
    @testset for T in [Int32, Int64, Float32, Float64]
        a = CuArray([zero(T)])

        function kernel(a, b, c)
            CUDA.atomic_cas!(pointer(a), b, c)
            return
        end

        @cuda threads=1024 kernel(a, zero(T), one(T))
        @test Array(a)[1] == one(T)
    end
end

end

@testset "atomic_max" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"6.0" && append!(types, [Float32, Float64])

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(a, T)
            i = threadIdx().x
            CUDA.atomic_max!(pointer(a), T(i))
            return
        end

        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == T(1024)
    end
end

@testset "atomic_min" begin
    types = [Int32, Int64, UInt32, UInt64]
    capability(device()) >= v"6.0" && append!(types, [Float32, Float64])

    @testset for T in types
        a = CuArray([T(1024)])

        function kernel(a, T)
            i = threadIdx().x
            CUDA.atomic_min!(pointer(a), T(i))
            return
        end

        @cuda threads=1024 kernel(a, T)
        @test Array(a)[1] == one(T)
    end
end

if capability(device()) >= v"6.0"

@testset "atomic_mul" begin
    @testset for T in [Float32, Float64]
        a = CuArray([one(T)])

        function kernel(a, b)
            CUDA.atomic_mul!(pointer(a), b)
            return
        end

        @cuda threads=10 kernel(a, T(2))
        @test Array(a)[1] == T(1024)
    end
end

@testset "atomic_div" begin
    @testset for T in [Float32, Float64]
        a = CuArray([T(1024)])

        function kernel(a, b)
            CUDA.atomic_div!(pointer(a), b)
            return
        end

        @cuda threads=10 kernel(a, T(2))
        @test Array(a)[1] == one(T)
    end
end

@testset "shared memory" begin
    function kernel()
        shared = @cuStaticSharedMem(Float32, 1)
        @atomic shared[threadIdx().x] += 0f0
        return
    end

    @cuda kernel()
    synchronize()
end

end

end

@testset "atomics (high-level)" begin

@testset "add" begin
    types = [Int32, Int64, UInt32, UInt64, Float32]
    capability(device()) >= v"6.0" && push!(types, Float64)

    @testset for T in types
        a = CuArray([zero(T)])

        function kernel(T, a)
            @atomic a[1] = a[1] + T(1)
            @atomic a[1] += T(1)
            return
        end

        @cuda threads=1024 kernel(T, a)
        @test Array(a)[1] == T(2048)
    end
end

@testset "sub" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([T(4096)])

        function kernel(T, a)
            @atomic a[1] = a[1] - T(1)
            @atomic a[1] -= T(1)
            return
        end

        @cuda threads=1024 kernel(T, a)
        @test Array(a)[1] == T(2048)
    end
end

@testset "and" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([~zero(T), ~zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = ~(T(1) << (i-1))
            @atomic a[1] = a[1] & mask
            @atomic a[2] &= mask
            return
        end

        @cuda threads=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == zero(T)
        @test Array(a)[2] == zero(T)
    end
end

@testset "or" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T), zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << (i-1)
            @atomic a[1] = a[1] | mask
            @atomic a[2] |= mask
            return
        end

        @cuda threads=8*sizeof(T) kernel(T, a)
        @test Array(a)[1] == ~zero(T)
        @test Array(a)[2] == ~zero(T)
    end
end

@testset "xor" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T), zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            mask = T(1) << ((i-1)%(8*sizeof(T)))
            @atomic a[1] = a[1] ⊻ mask
            @atomic a[2] ⊻= mask
            return
        end

        nb = 4
        @cuda threads=(8*sizeof(T)+nb) kernel(T, a)
        @test Array(a)[1] == ~zero(T) & ~((one(T) << nb) - one(T))
        @test Array(a)[2] == ~zero(T) & ~((one(T) << nb) - one(T))
    end
end

@testset "max" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([zero(T)])

        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] = max(a[1], T(i))
            return
        end

        @cuda threads=32 kernel(T, a)
        @test Array(a)[1] == 32
    end
end

@testset "min" begin
    @testset for T in [Int32, Int64, UInt32, UInt64]
        a = CuArray([typemax(T)])

        function kernel(T, a)
            i = threadIdx().x
            @atomic a[1] = min(a[1], T(i))
            return
        end

        @cuda threads=32 kernel(T, a)
        @test Array(a)[1] == 1
    end
end

@testset "macro" begin
    using CUDA: AtomicError

    @test_throws_macro AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = 1
    end
    @test_throws_macro AtomicError("right-hand side of an @atomic assignment should be a call") @macroexpand begin
        @atomic a[1] = b ? 1 : 2
    end

    @test_throws_macro AtomicError("right-hand side of a non-inplace @atomic assignment should reference the left-hand side") @macroexpand begin
        @atomic a[1] = a[2] + 1
    end

    @test_throws_macro AtomicError("unknown @atomic expression") @macroexpand begin
        @atomic wat(a[1])
    end

    @test_throws_macro AtomicError("@atomic should be applied to an array reference expression") @macroexpand begin
        @atomic a = a + 1
    end
end

end

############################################################################################

end

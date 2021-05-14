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
        CUDA.@sync @cuda kernel(x)
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

    test_output((1,), "(1,)")
    test_output((1,2), "(1, 2)")
    test_output((1,2,3.), "(1, 2, 3.000000)")


    # escaping

    kernel1(val) = (@cuprint(val); nothing)
    _, out = @grab_output @on_device kernel1(42)
    @test out == "42"

    kernel2(val) = (@cuprintln(val); nothing)
    _, out = @grab_output @on_device kernel2(42)
    @test out == "42$endline"
end

@testset "@cushow" begin
    function kernel()
        seven_i32 = Int32(7)
        three_f64 = Float64(3)
        @cushow seven_i32
        @cushow three_f64
        @cushow 1f0 + 4f0
        return
    end

    _, out = @grab_output @on_device kernel()
    @test out == "seven_i32 = 7$(endline)three_f64 = 3.000000$(endline)1.0f0 + 4.0f0 = 5.000000$(endline)"
end

@testset "@cushow array pointers" begin
    function kernel()
        a = @cuStaticSharedMem(Float32, 1)
        b = @cuStaticSharedMem(Float32, 2)
        @cushow pointer(a) pointer(b)
        return
    end

    _, out = @grab_output @on_device kernel()
    @test occursin("pointer(a) = ", out)
    @test occursin("pointer(b) = ", out)
    @test occursin("= 0x", out)
end

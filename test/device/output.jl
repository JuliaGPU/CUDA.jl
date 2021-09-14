
endline = Sys.iswindows() ? "\r\n" : "\n"

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
    function test_output(val)
        str = sprint(io->print(io, val))
        canary = rand(Int32) # if we mess up the main arg, this one will print wrong
        _, out = @grab_output @on_device @cuprint(val, " (", canary, ")")
        @test out == "$(str) ($(Int(canary)))"
    end

    for typ in (Int16, Int32, Int64, UInt16, UInt32, UInt64)
        test_output(typ(42))
    end

    for typ in (Float32, Float64)
        test_output(typ(42))
    end

    test_output(Cchar('c'))

    for typ in (Ptr{Cvoid}, Ptr{Int})
        ptr = convert(typ, Int(0x12345))
        test_output(ptr)
    end

    test_output(true)
    test_output(false)

    test_output((1,))
    test_output((1,2))
    test_output((1,2,3.))


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
    @test out == "seven_i32 = 7$(endline)three_f64 = 3.0$(endline)1.0f0 + 4.0f0 = 5.0$(endline)"
end

@testset "@cushow array pointers" begin
    function kernel()
        a = CuStaticSharedArray(Float32, 1)
        b = CuStaticSharedArray(Float32, 2)
        @cushow pointer(a) pointer(b)
        return
    end

    _, out = @grab_output @on_device kernel()
    @test occursin("pointer(a) = Core.LLVMPtr{Float32, 3}(0x0000000000000000)", out)
    @test occursin("pointer(b) = Core.LLVMPtr{Float32, 3}(0x0000000000000020)", out)
end

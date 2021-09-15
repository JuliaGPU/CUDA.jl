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

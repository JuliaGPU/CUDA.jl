# compiling algorithm instantiations with HostJIT

@testset "diagnostics" begin
    # the compiler's remarks about CUB's code don't end up on stderr
    keys = CuArray(rand(Int16, 1000))
    values = CuArray(rand(Int8, 1000))
    output = mktemp() do path, io
        redirect_stderr(io) do
            CCCL.radix_sort!(keys; values, rev=true)
        end
        read(path, String)
    end
    @test !occursin("generated when compiling", output)
    @test issorted(Array(keys); rev=true)

    # a failed build reports the compiler log; here ptxas rejects the inline assembly
    bogus(a) = (Base.llvmcall("""call void asm sideeffect "cccl.bogus;", ""()
                                 ret void""", Cvoid, Tuple{}); a)
    x = CuArray(rand(Float32, 1000))
    err = try
        CCCL.reduce((a, b) -> bogus(a) + b, x; init=0f0)
        nothing
    catch err
        err
    end
    @test err isa ErrorException
    @test occursin("Not a name of any known instruction: 'cccl'", sprint(showerror, err))
end

@testset "concurrent use" begin
    # tasks needing the same new build wait for one of them to create it
    x = CuArray(rand(Float32, 1000))
    op = (a, b) -> max(abs(a), abs(b))
    results = fetch.([Threads.@spawn CCCL.reduce(op, x; init=0f0) for _ in 1:4])
    @test all(==(maximum(Array(x))), results)
end

@testset "throwing operators" begin
    # CUB's kernels cannot report exceptions, so operators that throw are rejected
    x = CuArray(Float32[1, 2, -3, 4])
    err = try
        CCCL.reduce((a, b) -> a < 0 ? error("negative") : a + b, x; init=0f0)
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    @test occursin("cannot throw exceptions", sprint(showerror, err))
    @test occursin(basename(@__FILE__), sprint(showerror, err))

    t = (1f0, 2f0, 3f0)
    @test_throws ArgumentError CCCL.reduce((a, b) -> a + t[Int(b)], x; init=0f0)
    @test CCCL.reduce((a, b) -> a + b * @inbounds(t[1]), x; init=0f0) == 4f0
end

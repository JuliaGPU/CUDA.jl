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

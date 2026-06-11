@testset "error descriptions" begin
    # every result code must have a printable description; this guards against
    # referencing enum values that no longer exist in the headers (regression).
    for code in instances(cuFFT.cufftResult)
        err = cuFFT.CUFFTError(code)
        msg = sprint(showerror, err)
        @test occursin(string(code), msg)
        @test !isempty(cuFFT.description(err))
    end
end

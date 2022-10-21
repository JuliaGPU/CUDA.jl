using Test

# work around JuliaLang/Pkg.jl#2500
if VERSION < v"1.8"
    test_project = first(Base.load_path())
    preferences_file = joinpath(dirname(@__DIR__), "LocalPreferences.toml")
    test_preferences_file = joinpath(dirname(test_project), "LocalPreferences.toml")
    if isfile(preferences_file) && !isfile(test_preferences_file)
        cp(preferences_file, test_preferences_file)
    end
end

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using CUSTATEVEC
@test CUSTATEVEC.has_custatevec()
@info "CUSTATEVEC version: $(CUSTATEVEC.version())"

@testset "CUSTATEVEC" begin
    import CUSTATEVEC: CuStateVec, applyMatrix!, expectation, sample

    # build a simple state and compute expectations
    n_q = 2
    @testset for elty in [ComplexF32, ComplexF64]
        H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        X = convert(Matrix{elty}, [0 1; 1 0])
        Z = convert(Matrix{elty}, [1 0; 0 -1])
        sv = CuStateVec(elty, n_q)
        sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
        exp, res = expectation(sv, Z, Int32[0])
        @test exp ≈ 0.0 atol=1e-6
        exp, res = expectation(sv, X, Int32[0])
        @test exp ≈ 1.0 atol=1e-6
    end
    # build a simple state with controls and compute expectations
    n_q = 2
    @testset for elty in [ComplexF32, ComplexF64]
        H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        X = convert(Matrix{elty}, [0 1; 1 0])
        Z = convert(Matrix{elty}, [1 0; 0 -1])
        sv = CuStateVec(elty, n_q)
        sv = applyMatrix!(sv, H, false, Int32[0], Int32[])
        sv = applyMatrix!(sv, X, false, Int32[1], Int32[0]) # CNOT
        exp, res = expectation(sv, Z, Int32[0])
        @test exp ≈ 0.0 atol=1e-6
        exp, res = expectation(sv, X, Int32[0])
        @test exp ≈ 0.0 atol=1e-6
    end
    # build a simple state and compute samples
    n_q = 10 
    @testset for elty in [ComplexF32, ComplexF64]
        H = convert(Matrix{elty}, (1/√2).*[1 1; 1 -1])
        sv = CuStateVec(elty, n_q)
        for q in 0:n_q-1
            sv = applyMatrix!(sv, H, false, Int32[q], Int32[])
        end
        samples = sample(sv, Int32.(collect(0:n_q-1)), 10)
        # check that this succeeded
        @test length(samples) == 10
    end
end

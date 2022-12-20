using Test, LinearAlgebra

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
    import CUSTATEVEC: CuStateVec, applyMatrix!, applyPauliExp!, applyGeneralizedPermutationMatrix!, expectation, expectationsOnPauliBasis, sample, testMatrixType, Pauli, PauliX, PauliY, PauliZ, PauliI, measureOnZBasis!, swapIndexBits!

    @testset "applyMatrix! and expectation" begin
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
    end
    @testset "applyMatrix! and sample" begin
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
    @testset "applyPauliExp" begin
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
            h_sv_result = 1.0/√8 * elty[0.0, im, 0.0, -1.0, 0.0, im, 0.0, 1.0]
            sv   = CuStateVec(h_sv)
            sv   = applyPauliExp!(sv, π/2, [PauliZ()], Int32[2], Int32[1], Int32[1])
            sv_result = collect(sv.data)
            @test sv_result ≈ h_sv_result
        end
    end
    @testset "applyGeneralizedPermutationMatrix" begin
        @testset for elty in [ComplexF32, ComplexF64]
            diagonals   = elty[1.0, im, im, 1.0]
            permutation = [0, 2, 1, 3]
            h_sv        = elty[0.0, 0.1im, 0.1 + 0.1im, 0.1 + 0.2im, 0.2+0.2im, 0.3 + 0.3im, 0.3+0.4im, 0.4+0.5im]
            h_sv_result = elty[0.0, 0.1im, 0.1 + 0.1im, 0.1 + 0.2im, 0.2+0.2im, -0.4 + 0.3im, -0.3+0.3im, 0.4+0.5im]
            sv = CuStateVec{elty}(CuVector{elty}(h_sv), UInt32(log2(length(h_sv))))
            sv_result = applyGeneralizedPermutationMatrix!(sv, CuVector{Int64}(permutation), CuVector{elty}(diagonals), false, [0, 1], [2], [1])
            @test collect(sv_result.data) ≈ h_sv_result
        end
    end
    @testset "measureOnZBasis" begin
        @testset for elty in [ComplexF32, ComplexF64]
            h_sv = 1.0/√8 .* elty[0.0, im, 0.0, im, 0.0, im, 0.0, im]
            h_sv_result = 1.0/√2 * elty[0.0, 0.0, 0.0, im, 0.0, 0.0, 0.0, im]
            sv   = CuStateVec(h_sv)
            sv, parity = measureOnZBasis!(sv, [0, 1, 2], 0.2, CUSTATEVEC.CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO)
            sv_result  = collect(sv.data)
            @test sv_result ≈ h_sv_result
        end
    end
    @testset "swapIndexBits" begin
        @testset for elty in [ComplexF32, ComplexF64]
            # 0.1|000> + 0.4|011> - 0.4|101> - 0.3im|111>
            h_sv = elty[0.1, 0.0, 0.0, 0.4, 0.0, -0.4, 0.0, -0.3im]
            # 0.1|000> + 0.4|110> - 0.4|101> + 0.3im|111>
            h_sv_result = elty[0.1, 0.0, 0.0, 0.0, 0.0, -0.4, 0.4, -0.3im]
            sv   = CuStateVec(h_sv)
            swapped_sv = swapIndexBits!(sv, [0=>2], [1], [1])
            @test collect(swapped_sv.data) == h_sv_result
        end
    end
    @testset "testMatrixType $elty" for elty in [ComplexF32, ComplexF64]
        n = 128
        @testset "Hermitian matrix" begin
            A = rand(elty, n, n)
            A = A + A'
            @test testMatrixType(A, false, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(A, true, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), false, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), true, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
        end
        @testset "Unitary matrix" begin
            A = elty <: Real ? diagm(ones(elty, n)) : exp(im * 0.2 * diagm(ones(elty, n)))
            @test testMatrixType(A, false, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(A, true, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), false, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
            @test testMatrixType(CuMatrix{elty}(A), true, CUSTATEVEC.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
        end
    end
end

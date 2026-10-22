using cuSOLVER

if !cuSOLVER.has_cusolvermg()
@warn "CUSOLVERMG not available, skipping tests"
else

using LinearAlgebra
import LinearAlgebra: BlasInt

# the wrappers can be called from a device that is not part of the grid.
# test this first, before the grid below makes the pools of all devices accessible.
if ndevices() > 1
    @testset "calling device outside the grid" begin
        m = 256
        device!(last(collect(devices())))
        cuSOLVER.devices!([first(devices())])
        try
            for elty in [Float32, ComplexF64]
                A = rand(elty, m, m)
                A += A'
                W, _ = cuSOLVER.mg_syevd!('V', 'L', copy(A))
                @test W ≈ eigvals(Hermitian(A))

                A = rand(elty, m, m)
                F, ipiv = cuSOLVER.mg_getrf!(copy(A))
                alu = LinearAlgebra.LU(F, convert(Vector{BlasInt}, ipiv), zero(BlasInt))
                @test A ≈ Array(alu)
            end
        finally
            device!(first(devices()))
        end
    end
end

# use a grid of all available devices, or of the same device twice on single-GPU systems,
# so that the multi-device code paths are covered either way.
if ndevices() == 1
    cuSOLVER.devices!([device(), device()])
else
    cuSOLVER.devices!(collect(devices()))
end

# the wrappers split columns evenly across the grid, without a remainder
m = 128 * cuSOLVER.ndevices()

@testset "mg_syevd!" begin
    @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        A = rand(elty, m, m)
        A += A'
        hW = eigvals(Hermitian(A))
        hV = eigvecs(Hermitian(A))
        W, A = cuSOLVER.mg_syevd!('V','L',A)
        # compare
        @test W ≈ hW
        @test A*diagm(0=>W)*A' ≈ hV*diagm(0=>hW)*hV'

        A = rand(elty, m, m)
        A += A'
        hW = eigvals(Hermitian(A))
        W = cuSOLVER.mg_syevd!('N','L',A)
        # compare
        @test W ≈ hW

    end
end # elty

if cuSOLVER.version() >= v"10.4"
    @testset "mg_potrf!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A = rand(elty, m, m)
            A = A*A'+I # posdef
            hA = copy(A)
            A = cuSOLVER.mg_potrf!('L',A)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            # compare
            @test A ≈ hA
        end
    end # elty

    @testset "mg_potrf and mg_potri!" begin
        #@testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "element type $elty" for elty in [Float64, ComplexF64]
            A = rand(elty, m, m)
            A = A*A'+I # posdef
            hA = copy(A)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            A = cuSOLVER.mg_potrf!('L',A)
            LinearAlgebra.LAPACK.potri!('L', hA)
            A = cuSOLVER.mg_potri!('L',A)
            # compare
            @test tril(A) ≈ tril(hA)
        end
    end # elty

    @testset "mg_potrf and mg_potrs!" begin
        #@testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "element type $elty" for elty in [Float64, ComplexF64]
            A = rand(elty, m, m)
            B = rand(elty, m, m)
            A = A*A'+I # posdef
            hA = copy(A)
            hB = copy(B)
            LinearAlgebra.LAPACK.potrf!('L', hA)
            LinearAlgebra.LAPACK.potrs!('L', hA, hB)
            A = cuSOLVER.mg_potrf!('L', A)
            B = cuSOLVER.mg_potrs!('L', A, B)
            # compare
            tol    = real(elty) == Float32 ? 1e-4 : 1e-6
            @test A ≈ hA
            @test B ≈ hB rtol=tol
        end
    end # elty
end

if cuSOLVER.version() >= v"10.3"
    @testset "mg_getrf!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A      = rand(elty,m,m)
            h_A    = copy(A)
            A,ipiv = cuSOLVER.mg_getrf!(A)
            alu    = LinearAlgebra.LU(A, convert(Vector{BlasInt},ipiv), zero(BlasInt))
            @test h_A ≈ Array(alu)
        end
    end

    @testset "mg_getrs!" begin
        @testset "element type $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A      = rand(elty,m,m)
            h_A    = copy(A)
            alu    = lu(A, NoPivot())
            B      = rand(elty, m, div(m,2))
            h_B    = copy(B)
            tol    = real(elty) == Float32 ? 1e-1 : 1e-6
            B      = cuSOLVER.mg_getrs!('N', alu.factors, alu.ipiv, B)
            @test B ≈ h_A\h_B  rtol=tol
        end
    end
end

end

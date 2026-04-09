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
        @test testMatrixType(A, false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
        @test testMatrixType(A, true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
        @test testMatrixType(CuMatrix{elty}(A), false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
        @test testMatrixType(CuMatrix{elty}(A), true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_HERMITIAN) <= 200 * eps(real(elty))
    end
    @testset "Unitary matrix" begin
        A = elty <: Real ? diagm(ones(elty, n)) : exp(im * 0.2 * diagm(ones(elty, n)))
        @test testMatrixType(A, false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
        @test testMatrixType(A, true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
        @test testMatrixType(CuMatrix{elty}(A), false, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
        @test testMatrixType(CuMatrix{elty}(A), true, cuStateVec.CUSTATEVEC_MATRIX_TYPE_UNITARY) <= 200 * eps(real(elty))
    end
end
@testset "accessorSet!/accessorGet" begin
    nIndexBits = 3
    bitOrdering  = [1, 2, 0]
    @testset for elty in [ComplexF32, ComplexF64]
        h_sv = zeros(elty, 2^nIndexBits)
        h_sv_result = elty[0; 0.1im; 0.1+0.1im; 0.1+0.2im; 0.2+0.2im; 0.3+0.3im; 0.3+0.4im; 0.4+0.5im]
        buffer = elty[0; 0.1im; 0.1+0.1im; 0.1+0.2im; 0.2+0.2im; 0.3+0.3im; 0.3+0.4im; 0.4+0.5im]

        sv = CuStateVec(h_sv)
        acc = CuStateVecAccessor(sv, bitOrdering, Int[], Int[])
        accessorSet!(acc, buffer, 0, 2^nIndexBits)
        next_buf = similar(buffer)
        accessorGet(acc, next_buf, 0, 2^nIndexBits)
        @test next_buf == h_sv_result
    end
end

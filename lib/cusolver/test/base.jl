using cuSOLVER

@testset "CUSOLVER helpers and types" begin
    @test convert(cuSOLVER.cusolverEigType_t, 1) == cuSOLVER.CUSOLVER_EIG_TYPE_1
    @test convert(cuSOLVER.cusolverEigType_t, 2) == cuSOLVER.CUSOLVER_EIG_TYPE_2
    @test convert(cuSOLVER.cusolverEigType_t, 3) == cuSOLVER.CUSOLVER_EIG_TYPE_3
    @test_throws ArgumentError("Unknown eigenvalue solver type 4.") convert(cuSOLVER.cusolverEigType_t, 4)

    @test convert(cuSOLVER.cusolverEigMode_t, 'N') == cuSOLVER.CUSOLVER_EIG_MODE_NOVECTOR
    @test convert(cuSOLVER.cusolverEigMode_t, 'V') == cuSOLVER.CUSOLVER_EIG_MODE_VECTOR
    @test_throws ArgumentError("Unknown eigenvalue solver mode A.") convert(cuSOLVER.cusolverEigMode_t, 'A')
    
    @test convert(cuSOLVER.cusolverEigRange_t, 'A') == cuSOLVER.CUSOLVER_EIG_RANGE_ALL
    @test convert(cuSOLVER.cusolverEigRange_t, 'V') == cuSOLVER.CUSOLVER_EIG_RANGE_V
    @test convert(cuSOLVER.cusolverEigRange_t, 'I') == cuSOLVER.CUSOLVER_EIG_RANGE_I
    @test_throws ArgumentError("Unknown eigenvalue solver range B.") convert(cuSOLVER.cusolverEigRange_t, 'B')
    
    @test convert(cuSOLVER.cusolverStorevMode_t, 'C') == cuSOLVER.CUBLAS_STOREV_COLUMNWISE
    @test convert(cuSOLVER.cusolverStorevMode_t, 'R') == cuSOLVER.CUBLAS_STOREV_ROWWISE
    @test_throws ArgumentError("Unknown storage mode A.") convert(cuSOLVER.cusolverStorevMode_t, 'A')

    @test convert(cuSOLVER.cusolverDirectMode_t, 'F') == cuSOLVER.CUBLAS_DIRECT_FORWARD
    @test convert(cuSOLVER.cusolverDirectMode_t, 'B') == cuSOLVER.CUBLAS_DIRECT_BACKWARD
    @test_throws ArgumentError("Unknown direction mode A.") convert(cuSOLVER.cusolverDirectMode_t, 'A')

    @test convert(cuSOLVER.cusolverIRSRefinement_t, "NOT_SET")         == cuSOLVER.CUSOLVER_IRS_REFINE_NOT_SET
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "NONE")            == cuSOLVER.CUSOLVER_IRS_REFINE_NONE
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "CLASSICAL")       == cuSOLVER.CUSOLVER_IRS_REFINE_CLASSICAL
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "CLASSICAL_GMRES") == cuSOLVER.CUSOLVER_IRS_REFINE_CLASSICAL_GMRES
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "GMRES")           == cuSOLVER.CUSOLVER_IRS_REFINE_GMRES
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "GMRES_GMRES")     == cuSOLVER.CUSOLVER_IRS_REFINE_GMRES_GMRES
    @test convert(cuSOLVER.cusolverIRSRefinement_t, "GMRES_NOPCOND")   == cuSOLVER.CUSOLVER_IRS_REFINE_GMRES_NOPCOND
    @test_throws ArgumentError("Unknown iterative refinement solver A.") convert(cuSOLVER.cusolverIRSRefinement_t, "A")

    @test convert(cuSOLVER.cusolverPrecType_t, "R_16F")  == cuSOLVER.CUSOLVER_R_16F
    @test convert(cuSOLVER.cusolverPrecType_t, "R_16BF") == cuSOLVER.CUSOLVER_R_16BF
    @test convert(cuSOLVER.cusolverPrecType_t, "R_TF32") == cuSOLVER.CUSOLVER_R_TF32
    @test convert(cuSOLVER.cusolverPrecType_t, "C_16F")  == cuSOLVER.CUSOLVER_C_16F
    @test convert(cuSOLVER.cusolverPrecType_t, "C_16BF") == cuSOLVER.CUSOLVER_C_16BF
    @test convert(cuSOLVER.cusolverPrecType_t, "C_TF32") == cuSOLVER.CUSOLVER_C_TF32
    @test_throws ArgumentError("cusolverPrecType_t equivalent for input type A does not exist!") convert(cuSOLVER.cusolverPrecType_t, "A")
end

using CUDA.CUSOLVER

@testset "CUSOLVER helpers and types" begin
    @test convert(CUSOLVER.cusolverEigType_t, 1) == CUSOLVER.CUSOLVER_EIG_TYPE_1
    @test convert(CUSOLVER.cusolverEigType_t, 2) == CUSOLVER.CUSOLVER_EIG_TYPE_2
    @test convert(CUSOLVER.cusolverEigType_t, 3) == CUSOLVER.CUSOLVER_EIG_TYPE_3
    @test_throws ArgumentError("Unknown eigenvalue solver type 4.") convert(CUSOLVER.cusolverEigType_t, 4)

    @test convert(CUSOLVER.cusolverEigMode_t, 'N') == CUSOLVER.CUSOLVER_EIG_MODE_NOVECTOR
    @test convert(CUSOLVER.cusolverEigMode_t, 'V') == CUSOLVER.CUSOLVER_EIG_MODE_VECTOR
    @test_throws ArgumentError("Unknown eigenvalue solver mode A.") convert(CUSOLVER.cusolverEigMode_t, 'A')
    
    @test convert(CUSOLVER.cusolverEigRange_t, 'A') == CUSOLVER.CUSOLVER_EIG_RANGE_ALL
    @test convert(CUSOLVER.cusolverEigRange_t, 'V') == CUSOLVER.CUSOLVER_EIG_RANGE_V
    @test convert(CUSOLVER.cusolverEigRange_t, 'I') == CUSOLVER.CUSOLVER_EIG_RANGE_I
    @test_throws ArgumentError("Unknown eigenvalue solver range B.") convert(CUSOLVER.cusolverEigRange_t, 'B')
    
    @test convert(CUSOLVER.cusolverStorevMode_t, 'C') == CUSOLVER.CUBLAS_STOREV_COLUMNWISE
    @test convert(CUSOLVER.cusolverStorevMode_t, 'R') == CUSOLVER.CUBLAS_STOREV_ROWWISE
    @test_throws ArgumentError("Unknown storage mode A.") convert(CUSOLVER.cusolverStorevMode_t, 'A')

    @test convert(CUSOLVER.cusolverDirectMode_t, 'F') == CUSOLVER.CUBLAS_DIRECT_FORWARD
    @test convert(CUSOLVER.cusolverDirectMode_t, 'B') == CUSOLVER.CUBLAS_DIRECT_BACKWARD
    @test_throws ArgumentError("Unknown direction mode A.") convert(CUSOLVER.cusolverDirectMode_t, 'A')

    @test convert(CUSOLVER.cusolverIRSRefinement_t, "NOT_SET")         == CUSOLVER.CUSOLVER_IRS_REFINE_NOT_SET
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "NONE")            == CUSOLVER.CUSOLVER_IRS_REFINE_NONE
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "CLASSICAL")       == CUSOLVER.CUSOLVER_IRS_REFINE_CLASSICAL
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "CLASSICAL_GMRES") == CUSOLVER.CUSOLVER_IRS_REFINE_CLASSICAL_GMRES
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "GMRES")           == CUSOLVER.CUSOLVER_IRS_REFINE_GMRES
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "GMRES_GMRES")     == CUSOLVER.CUSOLVER_IRS_REFINE_GMRES_GMRES
    @test convert(CUSOLVER.cusolverIRSRefinement_t, "GMRES_NOPCOND")   == CUSOLVER.CUSOLVER_IRS_REFINE_GMRES_NOPCOND
    @test_throws ArgumentError("Unknown iterative refinement solver A.") convert(CUSOLVER.cusolverIRSRefinement_t, "A")

    @test convert(CUSOLVER.cusolverPrecType_t, "R_16F")  == CUSOLVER.CUSOLVER_R_16F
    @test convert(CUSOLVER.cusolverPrecType_t, "R_16BF") == CUSOLVER.CUSOLVER_R_16BF
    @test convert(CUSOLVER.cusolverPrecType_t, "R_TF32") == CUSOLVER.CUSOLVER_R_TF32
    @test convert(CUSOLVER.cusolverPrecType_t, "C_16F")  == CUSOLVER.CUSOLVER_C_16F
    @test convert(CUSOLVER.cusolverPrecType_t, "C_16BF") == CUSOLVER.CUSOLVER_C_16BF
    @test convert(CUSOLVER.cusolverPrecType_t, "C_TF32") == CUSOLVER.CUSOLVER_C_TF32
    @test_throws ArgumentError("cusolverPrecType_t equivalent for input type A does not exist!") convert(CUSOLVER.cusolverPrecType_t, "A")
end

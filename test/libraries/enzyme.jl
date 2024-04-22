using EnzymeCore
using GPUCompiler

@testset "compiler_job_from_backend" begin
    @test EnzymeCore.compiler_job_from_backend(CUDABackend(), typeof(()->nothing), Tuple{}) isa GPUCompiler.CompilerJob
end

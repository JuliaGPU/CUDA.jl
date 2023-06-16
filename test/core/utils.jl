using BFloat16s: BFloat16
import CUDA: cudaDataType

@testset "test utilities" begin
    mutable struct NoThrowTestSet <: Test.AbstractTestSet
        results::Vector
        NoThrowTestSet(desc) = new([])
    end
    Test.record(ts::NoThrowTestSet, t::Test.Result) = (push!(ts.results, t); t)
    Test.finish(ts::NoThrowTestSet) = ts.results
    fails = @testset NoThrowTestSet begin
        # OK
        @test_throws_cuerror CUDA.ERROR_UNKNOWN throw(CuError(CUDA.ERROR_UNKNOWN))
        # Fail, wrong CuError
        @test_throws_cuerror CUDA.ERROR_UNKNOWN throw(CuError(CUDA.ERROR_INVALID_VALUE))
        # Fail, wrong Exception
        @test_throws_cuerror CUDA.ERROR_UNKNOWN error()
    end
    @test isa(fails[1], Test.Pass)
    @test isa(fails[2], Test.Fail)
    @test isa(fails[3], Test.Fail)
end

@testset "@sync" begin
  t = Base.@elapsed ret = CUDA.@sync begin
    # TODO: do something that takes a while on the GPU
    #       (need to wrap clock64 for that)
    42
  end
  @test t >= 0
  @test ret == 42
end

@testset "versioninfo" begin
    CUDA.versioninfo(devnull)
end

# explicitly test these rather than just implicitly
# in library calls
@testset "cudaDataType conversions" begin
    for (j_type, c_type) in ((Float16, CUDA.R_16F), (Complex{Float16}, CUDA.C_16F), (BFloat16, CUDA.R_16BF), (Complex{BFloat16}, CUDA.C_16BF),
                             (Float32, CUDA.R_32F), (Complex{Float32}, CUDA.C_32F), (Float64, CUDA.R_64F), (Complex{Float64}, CUDA.C_64F),
                             (Int8, CUDA.R_8I), (Complex{Int8}, CUDA.C_8I), (UInt8, CUDA.R_8U), (Complex{UInt8}, CUDA.C_8U),
                             (Int16, CUDA.R_16I), (Complex{Int16}, CUDA.C_16I), (UInt16, CUDA.R_16U), (Complex{UInt16}, CUDA.C_16U),
                             (Int32, CUDA.R_32I), (Complex{Int32}, CUDA.C_32I), (UInt32, CUDA.R_32U), (Complex{UInt32}, CUDA.C_32U),
                             (Int64, CUDA.R_64I), (Complex{Int64}, CUDA.C_64I), (UInt64, CUDA.R_64U), (Complex{UInt64}, CUDA.C_64U))
        @test convert(cudaDataType, j_type) == c_type
        @test convert(Type, c_type) == j_type
    end
    @test_throws ArgumentError convert(cudaDataType, BigFloat)
    @test_throws ArgumentError convert(Type, CUDA.R_4I) # adjust once we support 4-bit Ints
end

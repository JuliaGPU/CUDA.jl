using BFloat16s: BFloat16
import CUDACore: cudaDataType

@testset "@sync" begin
  t = Base.@elapsed ret = CUDACore.@sync begin
    # TODO: do something that takes a while on the GPU
    #       (need to wrap clock64 for that)
    42
  end
  @test t >= 0
  @test ret == 42

  CUDACore.@sync blocking=true identity(nothing)
end

@testset "versioninfo" begin
    CUDACore.versioninfo(devnull)
end

# explicitly test these rather than just implicitly
# in library calls
@testset "cudaDataType conversions" begin
    for (j_type, c_type) in ((Float16, CUDACore.R_16F), (Complex{Float16}, CUDACore.C_16F), (BFloat16, CUDACore.R_16BF), (Complex{BFloat16}, CUDACore.C_16BF),
                             (Float32, CUDACore.R_32F), (Complex{Float32}, CUDACore.C_32F), (Float64, CUDACore.R_64F), (Complex{Float64}, CUDACore.C_64F),
                             (Int8, CUDACore.R_8I), (Complex{Int8}, CUDACore.C_8I), (UInt8, CUDACore.R_8U), (Complex{UInt8}, CUDACore.C_8U),
                             (Int16, CUDACore.R_16I), (Complex{Int16}, CUDACore.C_16I), (UInt16, CUDACore.R_16U), (Complex{UInt16}, CUDACore.C_16U),
                             (Int32, CUDACore.R_32I), (Complex{Int32}, CUDACore.C_32I), (UInt32, CUDACore.R_32U), (Complex{UInt32}, CUDACore.C_32U),
                             (Int64, CUDACore.R_64I), (Complex{Int64}, CUDACore.C_64I), (UInt64, CUDACore.R_64U), (Complex{UInt64}, CUDACore.C_64U))
        @test convert(cudaDataType, j_type) == c_type
        @test convert(Type, c_type) == j_type
    end
    @test_throws ArgumentError convert(cudaDataType, BigFloat)
    @test_throws ArgumentError convert(Type, CUDACore.R_4I) # adjust once we support 4-bit Ints
end

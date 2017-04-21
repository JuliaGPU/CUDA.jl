@testset "core infrastructure" begin

@testset "pointer" begin

# conversion to Ptr
@test_throws InexactError convert(Ptr{Void}, CU_NULL)
Base.unsafe_convert(Ptr{Void}, CU_NULL)

let
    @test eltype(DevicePtr{Void}) == Void
    @test eltype(CU_NULL) == Void
    @test isnull(CU_NULL)

    @test_throws InexactError convert(Ptr{Void}, CU_NULL)
    @test_throws InexactError convert(DevicePtr{Void}, C_NULL)
end

end


@testset "errors" begin

let
    ex = CuError(0)
    @test CUDAdrv.name(ex) == :SUCCESS
    @test CUDAdrv.description(ex) == "Success"
    
    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test contains(str, "0")
    @test contains(str, "Success")
end

let
    ex = CuError(0, "foobar")
    
    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test contains(str, "foobar")
end

end


@testset "base" begin

CUDAdrv.@apicall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

@test_throws ErrorException CUDAdrv.@apicall(:cuNonexisting, ())

@test_throws ErrorException @eval CUDAdrv.@apicall(:cuDummyAvailable, ())
@test_throws CUDAdrv.CuVersionError @eval CUDAdrv.@apicall(:cuDummyUnavailable, ())

@test_throws ErrorException eval(
    quote
        foo = :bar
        CUDAdrv.@apicall(foo, ())
    end
)

try
    CUDAdrv.@apicall(:cuDeviceGet, (Ptr{CUDAdrv.CuDevice_t}, Cint), Ref{CUDAdrv.CuDevice_t}(), devcount())
catch e
    e == CUDAdrv.ERROR_INVALID_DEVICE || rethrow(e)
end

CUDAdrv.vendor()

end

end
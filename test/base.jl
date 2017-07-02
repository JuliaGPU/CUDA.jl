@testset "base" begin

@test_throws ErrorException CUDAdrv.@apicall(:cuNonexisting, ())

@test_throws ErrorException @eval CUDAdrv.@apicall(:cuDummyAvailable, ())
@test_throws CUDAdrv.CuVersionError @eval CUDAdrv.@apicall(:cuDummyUnavailable, ())

@test_throws ErrorException eval(
    quote
        foo = :bar
        CUDAdrv.@apicall(foo, ())
    end
)

if CUDAdrv.configured
    @test_throws_cuerror CUDAdrv.ERROR_INVALID_DEVICE CUDAdrv.@apicall(:cuDeviceGet, (Ptr{CUDAdrv.CuDevice_t}, Cint), Ref{CUDAdrv.CuDevice_t}(), length(devices()))
end

CUDAdrv.vendor()

end

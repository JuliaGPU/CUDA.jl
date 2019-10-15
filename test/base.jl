@testset "base" begin

@test_throws LoadError @eval begin
    foo = :bar
    CUDAdrv.@apicall(foo, ())
end

@test_throws ErrorException CUDAdrv.@apicall(:cuNonexisting, ())

@test_throws ErrorException @eval CUDAdrv.@apicall(:cuDummyAvailable, ())
@test_throws CUDAdrv.CuVersionError @eval CUDAdrv.@apicall(:cuDummyUnavailable, ())

@test_throws_cuerror CUDAdrv.ERROR_INVALID_DEVICE CUDAdrv.@apicall(:cuDeviceGet, (Ptr{CUDAdrv.CUdevice}, Cint), Ref{CUDAdrv.CUdevice}(), length(devices()))

end

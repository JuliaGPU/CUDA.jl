@testset "pointer (on device)" begin

@testset "unsafe_load & unsafe_store!" begin

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64,
                   Float32,Float64),
             cached in (false, true)
    d_a = Mem.upload(ones(T))
    d_b = Mem.upload(zeros(T))

    ptr_a = CUDAnative.DevicePtr{T,AS.Global}(Base.unsafe_convert(Ptr{T}, d_a))
    ptr_b = CUDAnative.DevicePtr{T,AS.Global}(Base.unsafe_convert(Ptr{T}, d_b))

    @test Mem.download(T, d_a) != Mem.download(T, d_b)
    if cached
        @on_device Base.unsafe_store!($ptr_b, CUDAnative.unsafe_cached_load($ptr_a))
    else
        @on_device Base.unsafe_store!($ptr_b, CUDAnative.unsafe_load($ptr_a))
    end
    @test Mem.download(T, d_a) == Mem.download(T, d_b)
end

end

end

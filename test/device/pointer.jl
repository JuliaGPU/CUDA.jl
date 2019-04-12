@testset "pointer" begin

@testset "unsafe_load & unsafe_store!" begin

@eval struct LoadableStruct
    a::Int64
    b::UInt8
end
Base.one(::Type{LoadableStruct}) = LoadableStruct(1,1)
Base.zero(::Type{LoadableStruct}) = LoadableStruct(0,0)

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64, Int128,
                   Float32, Float64,
                   LoadableStruct),
             cached in (false, true)
    d_a = CuTestArray(ones(T))
    d_b = CuTestArray(zeros(T))

    ptr_a = CUDAnative.DevicePtr{T,AS.Global}(convert(CuPtr{T}, d_a.buf))
    ptr_b = CUDAnative.DevicePtr{T,AS.Global}(convert(CuPtr{T}, d_b.buf))
    @test Array(d_a) != Array(d_b)

    let ptr_a=ptr_a, ptr_b=ptr_b #JuliaLang/julia#15276
        if cached && capability(dev) >= v"3.2"
            @on_device unsafe_store!(ptr_b, unsafe_cached_load(ptr_a))
        else
            @on_device unsafe_store!(ptr_b, unsafe_load(ptr_a))
        end
    end
    @test Array(d_a) == Array(d_b)
end

@testset "indexing" begin
    function kernel(src, dst)
        unsafe_store!(dst, CUDAnative.unsafe_cached_load(src, 4))
        return
    end

    T = Complex{Int8}

    src = CuTestArray([T(1) T(9); T(3) T(4)])
    dst = CuTestArray([0])

    @cuda kernel(
        CUDAnative.DevicePtr{T,AS.Global}(convert(CuPtr{T}, src.buf)),
        CUDAnative.DevicePtr{T,AS.Global}(convert(CuPtr{T}, dst.buf))
    )

    @test Array(src)[4] == Array(dst)[1]
end

end

end

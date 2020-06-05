@inline function calcpoint(blockIdx, blockDim, threadIdx, size)
    i = (blockIdx - 1) * blockDim + threadIdx
    return i, Float32(i)
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,1}, texture::CuDeviceTexture{T,1}) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    @inbounds dst[i] = texture[u];
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,2}, texture::CuDeviceTexture{T,2}) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    @inbounds dst[i,j] = texture[u,v];
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{T,3}, texture::CuDeviceTexture{T,3}) where {T}
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    k, w = calcpoint(blockIdx().z, blockDim().z, threadIdx().z, size(dst)[3])
    @inbounds dst[i,j,k] = texture[u,v,w];
    return nothing
end

function fetch_all(texture)
    dims = size(texture)
    d_out = CuArray{eltype(texture)}(undef, dims...)
    @cuda threads = dims kernel_texture_warp_native(d_out, texture)
    d_out
end

@testset "Using CuTextureArray initialized from device" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
    for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end
    d_a1D = CuArray(a1D)
    d_a2D = CuArray(a2D)
    d_a3D = CuArray(a3D)

    texarr1D = CuTextureArray{Float32}(testheight)
    copyto!(texarr1D, d_a1D)
    tex1D = CuTexture(texarr1D)
    @test fetch_all(tex1D) == d_a1D

    texarr2D = CuTextureArray{Float32}(testheight, testwidth)
    tex2D = CuTexture(texarr2D)
    copyto!(texarr2D, d_a2D)
    @test fetch_all(tex2D) == d_a2D

    texarr2D_dir = CuTextureArray(d_a2D)
    tex2D_dir = CuTexture(texarr2D_dir)
    @test fetch_all(tex2D_dir) == d_a2D

    texarr3D = CuTextureArray{Float32}(testheight, testwidth, testdepth)
    tex3D = CuTexture(texarr3D)
    copyto!(texarr3D, d_a3D)
    @test fetch_all(tex3D) == d_a3D

    tex3D_direct = CuTexture{Float32}(testheight, testwidth, testdepth)
    copyto!(tex3D_direct.mem, d_a3D)
    @test fetch_all(tex3D_direct) == d_a3D
end

@testset "Using CuTextureArray initialized from host" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
    for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end

    texarr1D = CuTextureArray{Float32}(testheight)
    copyto!(texarr1D, a1D)
    tex1D = CuTexture(texarr1D)
    @test Array(fetch_all(tex1D)) == a1D

    texarr2D = CuTextureArray{Float32}(testheight, testwidth)
    copyto!(texarr2D, a2D)
    tex2D = CuTexture(texarr2D)
    @test Array(fetch_all(tex2D)) == a2D

    tex2D_dir = CuTexture(CuTextureArray(a2D))
    @test Array(fetch_all(tex2D_dir)) == a2D

    texarr3D = CuTextureArray{Float32}(testheight, testwidth, testdepth)
    copyto!(texarr3D, a3D)
    tex3D = CuTexture(texarr3D)
    @test Array(fetch_all(tex3D)) == a3D
end

@testset "Wrapping CuArray" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    d_a1D = CuArray(a1D)
    d_a2D = CuArray(a2D)

    # Strangely, this is not working
    texwrap1D = CuTexture(d_a1D)
    @test_broken fetch_all(texwrap1D) == d_a1D

    # This works as long as d_a2D is well pitched
    texwrap2D = CuTexture(d_a2D)
    @test fetch_all(texwrap2D) == d_a2D
end

@testset "All CUDA types" begin
    for T in (Int32, UInt32, Int16, UInt16, Int8, UInt8, Float32, Float16)
        testheight, testwidth, testdepth = 32, 32, 4
        a2D = rand(T, testheight, testwidth)
        d_a2D = CuArray(a2D)

        # Using CuTextureArray
        tex_2D = CuTexture{T}(testheight, testwidth)
        copyto!(tex_2D.mem, d_a2D)
        @test fetch_all(tex_2D) == d_a2D

        # Wrapping CuArray
        # This works as long as d_a2D is well pitched
        texwrap_2D = CuTexture(d_a2D)
        @test fetch_all(texwrap_2D) == d_a2D
    end
end

@testset "Multiple channels" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a2D = [(Int32(i), Int32(j)) for i = 1:testheight, j = 1:testwidth]
    d_a2D = CuArray(a2D)
    texarr2D = CuTextureArray{eltype(d_a2D)}(size(d_a2D)...)
    copyto!(texarr2D, d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D

    testheight, testwidth, testdepth = 16, 16, 4
    a2D = [(Int16(i), Int16(j), Int16(i + j), Int16(i - j)) for i = 1:testheight, j = 1:testwidth]
    d_a2D = CuArray(a2D)
    texarr2D = CuTextureArray{eltype(d_a2D)}(size(d_a2D)...)
    copyto!(texarr2D, d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D
end

@testset "Custom type" begin
    @testset "Auto cast" begin
        struct AKindOfRGBA
            r::UInt8
            g::UInt8
            b::UInt8
            a::UInt8
        end

        @test cuda_texture_alias_type(AKindOfRGBA) == NTuple{4,UInt8}

        testheight, testwidth, testdepth = 16, 16, 4
        a2D = [AKindOfRGBA(UInt8(i), UInt8(j), UInt8(j + i), UInt8(j > i)) for i = 1:testheight, j = 1:testwidth]
        d_a2D = CuArray(a2D)

        texarr2D = CuTextureArray{eltype(d_a2D)}(size(d_a2D)...)
        copyto!(texarr2D, d_a2D)
        tex2D = CuTexture(texarr2D)

        @test fetch_all(tex2D) == d_a2D
    end

    @testset "Manual cast" begin
        primitive type AKindOfFloat32 32 end

        # It is not enough to define `cuda_texture_alias_type(::Type{AKindOfFloat32}) = Float32`, one has to define the whole type.
        #   This is due to `cuda_texture_alias_type` being a *`@generated`* function.
        CUDA.cuda_texture_alias_type(::Type{NTuple{2,AKindOfFloat32}}) = NTuple{2,Float32}
        @test cuda_texture_alias_type(NTuple{2,AKindOfFloat32}) == NTuple{2,Float32}

        testheight, testwidth, testdepth = 16, 16, 4
        a2D = [
            (reinterpret(AKindOfFloat32, Float32(i)), reinterpret(AKindOfFloat32, Float32(j)))
            for i = 1:testheight, j = 1:testwidth]
        d_a2D = CuArray(a2D)

        texarr2D = CuTextureArray{eltype(d_a2D)}(size(d_a2D)...)
        copyto!(texarr2D, d_a2D)
        tex2D = CuTexture(texarr2D)

        @test fetch_all(tex2D) == d_a2D
    end
end

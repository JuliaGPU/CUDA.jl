@inline function calcpoint(blockIdx, blockDim, threadIdx, size)
    i = (blockIdx - 1) * blockDim + threadIdx
    return i, Float32(i)
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,1}, texture::CuDeviceTexture{<:Any,1})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    @inbounds dst[i] = texture[u]
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,2}, texture::CuDeviceTexture{<:Any,2})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    @inbounds dst[i,j] = texture[u,v]
    return nothing
end
function kernel_texture_warp_native(dst::CuDeviceArray{<:Any,3}, texture::CuDeviceTexture{<:Any,3})
    i, u = calcpoint(blockIdx().x, blockDim().x, threadIdx().x, size(dst)[1])
    j, v = calcpoint(blockIdx().y, blockDim().y, threadIdx().y, size(dst)[2])
    k, w = calcpoint(blockIdx().z, blockDim().z, threadIdx().z, size(dst)[3])
    @inbounds dst[i,j,k] = texture[u,v,w]
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

    texarr1D = CuTextureArray(d_a1D)
    tex1D = CuTexture(texarr1D)
    @test fetch_all(tex1D) == d_a1D

    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D

    texarr3D = CuTextureArray(d_a3D)
    tex3D = CuTexture(texarr3D)
    @test fetch_all(tex3D) == d_a3D
end

@testset "Using CuTextureArray initialized from host" begin
    testheight, testwidth, testdepth = 16, 16, 4
    a1D = convert(Array{Float32}, 1:testheight)
    a2D = convert(Array{Float32}, repeat(1:testheight, 1, testwidth) + repeat(0.01 * (1:testwidth)', testheight, 1))
    a3D = convert(Array{Float32}, repeat(a2D, 1, 1, testdepth))
    for k = 1:testdepth; a3D[:,:,k] .+= 0.0001 * k; end

    texarr1D = CuTextureArray(a1D)
    copyto!(texarr1D, a1D)
    tex1D = CuTexture(texarr1D)
    @test Array(fetch_all(tex1D)) == a1D

    texarr2D = CuTextureArray(a2D)
    tex2D = CuTexture(texarr2D)
    @test Array(fetch_all(tex2D)) == a2D

    tex2D_dir = CuTexture(CuTextureArray(a2D))
    @test Array(fetch_all(tex2D_dir)) == a2D

    texarr3D = CuTextureArray(a3D)
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
        tex_2D = CuTexture(d_a2D)
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
    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D

    testheight, testwidth, testdepth = 16, 16, 4
    a2D = [(Int16(i), Int16(j), Int16(i + j), Int16(i - j)) for i = 1:testheight, j = 1:testwidth]
    d_a2D = CuArray(a2D)
    texarr2D = CuTextureArray(d_a2D)
    tex2D = CuTexture(texarr2D)
    @test fetch_all(tex2D) == d_a2D
end

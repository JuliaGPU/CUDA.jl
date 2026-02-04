@testset "ldg" begin
    ir = sprint(io->CUDA.code_llvm(io, CUDA.pointerref_ldg, Tuple{Core.LLVMPtr{Int,AS.Global},Int,Val{1}}))
    if Base.libllvm_version >= v"20"
        @test occursin("load i64, ptr addrspace(1)", ir)
    else
        # `@llvm.nvvm.ldg` only exists in LLVM <20
        @test occursin("@llvm.nvvm.ldg", ir)
    end
end


capability(device()) >= v"3.2" && @testset "unsafe_cached_load" begin

@testset for T in (Int8, UInt16, Int32, UInt32, Int64, UInt64, Int128, Float32, Float64)
    d_a = CuArray(ones(T))
    d_b = CuArray(zeros(T))
    @test Array(d_a) != Array(d_b)

    ptr_a = reinterpret(Core.LLVMPtr{T,AS.Global}, pointer(d_a))
    ptr_b = reinterpret(Core.LLVMPtr{T,AS.Global}, pointer(d_b))

    let ptr_a=ptr_a, ptr_b=ptr_b #JuliaLang/julia#15276
        @on_device unsafe_store!(ptr_b, unsafe_cached_load(ptr_a))
    end

    @test Array(d_a) == Array(d_b)
end

@testset for (N, T) in ((4, Float32), (2, Float64), (4, Int8), (4, Int16), (4, Int32), (2, Int64))
    d_a = CuArray(ones(T, N))
    d_b = CuArray(zeros(T, N))
    @test Array(d_a) != Array(d_b)

    ptr_a = reinterpret(Core.LLVMPtr{NTuple{N, Base.VecElement{T}},AS.Global}, pointer(d_a))
    ptr_b = reinterpret(Core.LLVMPtr{NTuple{N, Base.VecElement{T}},AS.Global}, pointer(d_b))

    let ptr_a=ptr_a, ptr_b=ptr_b #JuliaLang/julia#15276
        @on_device unsafe_store!(ptr_b, unsafe_cached_load(ptr_a, 1, Val(16)), 1, Val(16))
    end

    @test Array(d_a) == Array(d_b)
end

@testset "Const" begin
    function kernel(a, b, i)
        @inbounds b[i] = Base.Experimental.Const(a)[i]
        return
    end

    buf = IOBuffer()

    a = CuArray([0])
    b = CuArray([0])
    @device_code_ptx io=buf @cuda kernel(a, b, 1)
    @test Array(a) == Array(b)

    asm = String(take!(copy(buf)))
    @test occursin("ld.global.nc", asm)


    function copy_const(A, _B)
        B = Base.Experimental.Const(_B)
        i = threadIdx().x
        if i <= length(A)
            @inbounds A[i] = B[i]
        end
        return
    end

    x = CUDA.zeros(Float64, 32)
    y = CUDA.ones(Float64, length(x))

    @cuda threads=length(x) copy_const(x, y)
    @test Array(x) == Array(y)
end

@testset "Const Vectorized" begin
    function kernel(a, b, i)
        ptr_a = reinterpret(Core.LLVMPtr{NTuple{4, Base.VecElement{Float32}},AS.Global}, pointer(a))
        ptr_b = reinterpret(Core.LLVMPtr{NTuple{4, Base.VecElement{Float32}},AS.Global}, pointer(b))
        unsafe_store!(ptr_b, unsafe_cached_load(ptr_a, i, Val(16)), i, Val(16))
        return
    end

    buf = IOBuffer()

    a = CUDA.ones(Float32, 4)
    b = CUDA.zeros(Float32, 4)
    @device_code_ptx io=buf @cuda kernel(a, b, 1)
    @test Array(a) == Array(b)

    asm = String(take!(copy(buf)))
    @test occursin("ld.global.nc.v4", asm)
    @test occursin("st.global.v4", asm)

    function kernel_const(a, b, i)
        @inbounds b[i] = Base.Experimental.Const(a)[i]
        return
    end

    buf = IOBuffer()

    a = CuArray([NTuple{4, Base.VecElement{Float32}}((1, 1, 1, 1))])
    b = CuArray([NTuple{4, Base.VecElement{Float32}}((0, 0, 0, 0))])
    @device_code_ptx io=buf @cuda kernel(a, b, 1)
    @test Array(a) == Array(b)

    asm = String(take!(copy(buf)))
    @test occursin("ld.global.nc.v4", asm)
    @test occursin("st.global.v4", asm)

    function copy_const(A, _B)
        B = Base.Experimental.Const(_B)
        i = threadIdx().x
        if i <= length(A)
            @inbounds A[i] = B[i]
        end
        return
    end

    x = CuArray([NTuple{4, Base.VecElement{Float32}}((0, 0, 0, 0)) for _ in 1:8])
    y = CuArray([NTuple{4, Base.VecElement{Float32}}((1, 1, 1, 1)) for _ in eachindex(x)])

    @cuda threads=length(x) copy_const(x, y)
    @test Array(x) == Array(y)
end

end

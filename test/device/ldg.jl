@testset "ldg" begin
    ir = sprint(io->CUDA.code_llvm(io, CUDA.pointerref_ldg, Tuple{Core.LLVMPtr{Int,AS.Global},Int,Val{1}}))
    @test occursin("@llvm.nvvm.ldg", ir)
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
end

end

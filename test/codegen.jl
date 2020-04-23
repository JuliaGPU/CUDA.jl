@testset "code generation" begin

############################################################################################

@testset "LLVM IR" begin

@testset "JuliaLang/julia#21121" begin
    function foobar()
        weight_matrix = @cuStaticSharedMem(Float32, (16, 16))
        sync_threads()
        weight_matrix[1, 16] *= 2
        sync_threads()
    end

    ir = sprint(io->CUDAnative.code_llvm(io, foobar, Tuple{}))
    @test !occursin("inttoptr", ir)
end

@testset "PTX TBAA" begin
    load(ptr) = unsafe_load(ptr)
    store(ptr) = unsafe_store!(ptr, 0)

    for f in (load, store)
        ir = sprint(io->CUDAnative.code_llvm(io, f,
                                             Tuple{CUDAnative.DevicePtr{Float32,AS.Global}};
                                             dump_module=true, raw=true))
        @test occursin("gputbaa_global", ir)

        # no TBAA on generic pointers
        ir = sprint(io->CUDAnative.code_llvm(io, f,
                                             Tuple{CUDAnative.DevicePtr{Float32,AS.Generic}};
                                             dump_module=true, raw=true))
        @test !occursin("gputbaa", ir)
    end


    cached_load(ptr) = unsafe_cached_load(ptr)

    ir = sprint(io->CUDAnative.code_llvm(io, cached_load,
                                         Tuple{CUDAnative.DevicePtr{Float32,AS.Global}};
                                         dump_module=true, raw=true))
    @test occursin("gputbaa_global", ir)
end

@testset "ghost values" begin
    @eval struct Singleton end

    ir = sprint(io->CUDAnative.code_llvm(io, unsafe_load,
                                         Tuple{CUDAnative.DevicePtr{Singleton,AS.Global}}))
    @test occursin("ret void", ir)
    @test unsafe_load(reinterpret(CUDAnative.DevicePtr{Singleton,AS.Global}, C_NULL)) === Singleton()

    ir = sprint(io->CUDAnative.code_llvm(io, unsafe_store!,
                                         Tuple{CUDAnative.DevicePtr{Singleton,AS.Global},
                                         Singleton}))
    @test !occursin("\bstore\b", ir)
end

@testset "CUDAnative.jl#553" begin
    function kernel(ptr)
       unsafe_store!(ptr, CUDAnative.fma(unsafe_load(ptr), unsafe_load(ptr,2), unsafe_load(ptr,3)))
       return
    end

    ir = sprint(io->CUDAnative.code_llvm(io, kernel, Tuple{Ptr{Float32}}))
    @test !occursin("@__nv_fmaf", ir)
end

@testset "reinterpret(Nothing, nothing)" begin
    kernel(ptr) = Base.unsafe_load(ptr)
    CUDAnative.code_llvm(devnull, kernel, Tuple{CUDAnative.DevicePtr{Nothing,CUDAnative.AS.Global}}; strict=true)
end

@testset "ldg" begin
    ir = sprint(io->CUDAnative.code_llvm(io, CUDAnative.pointerref_ldg, Tuple{CUDAnative.DevicePtr{Int,CUDAnative.AS.Global},Int,Val{1}}))
    @test occursin("@llvm.nvvm.ldg", ir)
end

@testset "assume" begin
    foo(i) = cld(42, i)
    ir = sprint(io->CUDAnative.code_llvm(io, foo, Tuple{Int}))
    @test occursin("@gpu_report_exception", ir)


    bar(i) = (CUDAnative.assume(i > 0); cld(42, i))
    ir = sprint(io->CUDAnative.code_llvm(io, bar, Tuple{Int}))
    @test !occursin("gpu_report_exception", ir)
end

end

############################################################################################

end

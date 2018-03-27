@testset "code generation" begin

############################################################################################

@testset "LLVM IR" begin

@testset "basic reflection" begin
    @eval llvm_valid_kernel() = nothing
    @eval llvm_invalid_kernel() = 1

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{}; optimize=false, dump_module=true))

    # module should contain our function + a generic call wrapper
    @test occursin("define void @julia_llvm_valid_kernel", ir)
    @test !occursin("define %jl_value_t* @jlcall_", ir)

    @test CUDAnative.code_llvm(devnull, llvm_invalid_kernel, Tuple{}) == nothing
    @test_throws ArgumentError CUDAnative.code_llvm(devnull, llvm_invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "exceptions" begin
    @eval codegen_exception() = throw(DivideError())
    ir = sprint(io->CUDAnative.code_llvm(io, codegen_exception, Tuple{}))

    # exceptions should get lowered to a plain trap...
    @test occursin("llvm.trap", ir)
    # not a jl_throw referencing a jl_value_t representing the exception
    @test !occursin("jl_value_t", ir)
    @test !occursin("jl_throw", ir)
end

@testset "sysimg" begin
    # bug: use a system image function

    @eval function codegen_call_sysimg(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
        return
    end

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_call_sysimg, Tuple{Ptr{Int},Int}))
    @test !occursin("jlsys_", ir)
end

@testset "child functions" begin
    # we often test using `@noinline sink` child functions, so test whether these survive
    @eval @noinline codegen_child(i) = (sink(i); nothing)
    @eval codegen_parent(i) = codegen_child(i)

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_parent, Tuple{Int}))
    @test occursin(r"call .+ @julia_codegen_child_", ir)
end

@testset "JuliaLang/julia#21121" begin
    @eval function codegen_tuple_leak()
        weight_matrix = @cuStaticSharedMem(Float32, (16, 16))
        sync_threads()
        weight_matrix[1, 16] *= 2
        sync_threads()

        return
    end

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_tuple_leak, Tuple{}))
    @test !occursin("inttoptr", ir)
end

@testset "kernel functions" begin
@testset "wrapper function aggregate rewriting" begin
    @eval codegen_aggregates(x) = nothing

    @eval struct Aggregate
        x::Int
    end

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_aggregates, Tuple{Aggregate}))
    @test occursin(Regex("@julia_codegen_aggregates_\\d+\\({ i64 }( addrspace\\(\\d+\\))?\\*"), ir)

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_aggregates, Tuple{Aggregate}; kernel=true))
    @test occursin(Regex("@ptxcall_codegen_aggregates_\\d+\\({ i64 }\\)"), ir)
end

@testset "property_annotations" begin
    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{}; dump_module=true))
    @test !occursin("nvvm.annotations", ir)

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{};
                                         dump_module=true, kernel=true))
    @test occursin("nvvm.annotations", ir)
    @test !occursin("maxntid", ir)
    @test !occursin("reqntid", ir)
    @test !occursin("minctasm", ir)
    @test !occursin("maxnreg", ir)

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{};
                                         dump_module=true, kernel=true, maxthreads=42))
    @test occursin("maxntidx\", i32 42", ir)
    @test occursin("maxntidy\", i32 1", ir)
    @test occursin("maxntidz\", i32 1", ir)

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{};
                                         dump_module=true, kernel=true, minthreads=42))
    @test occursin("reqntidx\", i32 42", ir)
    @test occursin("reqntidy\", i32 1", ir)
    @test occursin("reqntidz\", i32 1", ir)

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{};
                                         dump_module=true, kernel=true, blocks_per_sm=42))
    @test occursin("minctasm\", i32 42", ir)

    ir = sprint(io->CUDAnative.code_llvm(io, llvm_valid_kernel, Tuple{};
                                         dump_module=true, kernel=true, maxregs=42))
    @test occursin("maxnreg\", i32 42", ir)
end
end

@testset "LLVM D32593" begin
    @eval struct llvm_D32593_struct
        foo::Float32
        bar::Float32
    end

    @eval llvm_D32593(arr) = arr[1].foo

    CUDAnative.code_llvm(devnull, llvm_D32593, Tuple{CuDeviceVector{llvm_D32593_struct,AS.Global}})
end

@testset "julia calling convention" begin
    @eval codegen_specsig_va(Is...) = nothing
    @test_throws ArgumentError CUDAnative.code_llvm(devnull, codegen_specsig_va, Tuple{})
end

end


############################################################################################

@testset "PTX assembly" begin

@testset "basic reflection" begin
    @eval ptx_valid_kernel() = nothing
    @eval ptx_invalid_kernel() = 1

    @test CUDAnative.code_ptx(devnull, ptx_valid_kernel, Tuple{}) == nothing
    @test CUDAnative.code_ptx(devnull, ptx_invalid_kernel, Tuple{}) == nothing
    @test_throws ArgumentError CUDAnative.code_ptx(devnull, ptx_invalid_kernel, Tuple{}; kernel=true) == nothing
end

@testset "child functions" begin
    # we often test using @noinline child functions, so test whether these survive
    # (despite not having side-effects)
    @eval @noinline ptx_child(i) = (sink(i); nothing)
    @eval ptx_parent(i) = ptx_child(i)

    asm = sprint(io->CUDAnative.code_ptx(io, ptx_parent, Tuple{Int64}))
    @test occursin(r"call.uni\s+julia_ptx_child_"m, asm)
end

@testset "kernel functions" begin
    @eval @noinline ptx_nonentry(i) = (sink(i); nothing)
    @eval ptx_entry(i) = ptx_nonentry(i)

    asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64}; kernel=true))
    @test occursin(r"\.visible \.entry ptxcall_ptx_entry_", asm)
    @test !occursin(r"\.visible \.func julia_ptx_nonentry_", asm)
    @test occursin(r"\.func julia_ptx_nonentry_", asm)

@testset "property_annotations" begin
    asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64}; kernel=true))
    @test !occursin("maxntid", asm)

    asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64};
                                         kernel=true, maxthreads=42))
    @test occursin(".maxntid 42, 1, 1", asm)

    asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64};
                                         kernel=true, minthreads=42))
    @test occursin(".reqntid 42, 1, 1", asm)

    asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64};
                                         kernel=true, blocks_per_sm=42))
    @test occursin(".minnctapersm 42", asm)

    if LLVM.version() >= v"4.0"
        asm = sprint(io->CUDAnative.code_ptx(io, ptx_entry, Tuple{Int64};
                                             kernel=true, maxregs=42))
        @test occursin(".maxnreg 42", asm)
    end
end
end

@testset "IR validation" begin
@info "The next couple of warnings are expected"
@testset "delayed lookup" begin
    @eval codegen_ref_nonexisting() = nonexisting
    @test_throws ErrorException CUDAnative.code_ptx(codegen_ref_nonexisting, Tuple{})
end

@testset "generic call" begin
    @eval codegen_call_nonexisting() = nonexisting()
    @test_throws ErrorException CUDAnative.code_ptx(codegen_call_nonexisting, Tuple{})
end
end

@testset "idempotency" begin
    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)

    @eval codegen_idempotency() = nothing
    CUDAnative.code_ptx(devnull, codegen_idempotency, Tuple{})
    CUDAnative.code_ptx(devnull, codegen_idempotency, Tuple{})
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    @eval @noinline codegen_child_reuse_child(i) = (sink(i); nothing)
    @eval function codegen_child_reuse_parent1(i)
        codegen_child_reuse_child(i)
        return
    end

    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_parent1, Tuple{Int}))
    @test occursin(r".func julia_codegen_child_reuse_child_", asm)

    @eval function codegen_child_reuse_parent2(i)
        codegen_child_reuse_child(i+1)
        return
    end

    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_parent2, Tuple{Int}))
    @test occursin(r".func julia_codegen_child_reuse_child_", asm)
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions
    @eval @noinline codegen_child_reuse_bis_child1(i) = sink(i)
    @eval @noinline codegen_child_reuse_bis_child2(i) = sink(i+1)
    @eval function codegen_child_reuse_bis_parent1(i)
        codegen_child_reuse_bis_child1(i) + codegen_child_reuse_bis_child2(i)
        return
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_bis_parent1, Tuple{Int}))

    @eval function codegen_child_reuse_bis_parent2(i)
        codegen_child_reuse_bis_child1(i+1) + codegen_child_reuse_bis_child2(i+1)
        return
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_bis_parent2, Tuple{Int}))
end

@testset "indirect sysimg function use" begin
    # issue #9: re-using sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws, so the PTX code shouldn't contain a throw)

    # FIXME: Int64 because of #49

    @eval function codegen_recompile(out)
        wid, lane = fldmod1(unsafe_load(out), Int64(32))
        unsafe_store!(out, wid)
        return
    end

    asm = sprint(io->CUDAnative.code_ptx(io, codegen_recompile, Tuple{Ptr{Int64}}))
    @test !occursin("jl_throw", asm)
    @test !occursin("jl_invoke", asm)   # forced recompilation should still not invoke
end

@testset "compile for host after PTX" begin
    # issue #11: re-using host functions after PTX compilation
    @eval @noinline codegen_recompile_bis_child(i) = sink(i+1)

    @eval function codegen_recompile_bis_fromhost()
        codegen_recompile_bis_child(10)
    end

    @eval function codegen_recompile_bis_fromptx()
        codegen_recompile_bis_child(10)
        return
    end

    CUDAnative.code_ptx(devnull, codegen_recompile_bis_fromptx, Tuple{})
    @test codegen_recompile_bis_fromhost() == 11
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    @eval codegen_issue_13(x) = unsafe_trunc(Int, x)
    CUDAnative.code_ptx(devnull, codegen_issue_13, Tuple{Float64})
end

end


############################################################################################

end

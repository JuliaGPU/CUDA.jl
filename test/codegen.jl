@testset "code generation" begin

############################################################################################

@testset "LLVM IR" begin
    foo() = return nothing
    ir = sprint(io->CUDAnative.code_llvm(io, foo, (); optimize=false, dump_module=true))

    # module should contain our function + a generic call wrapper
    @test contains(ir, "define void @julia_foo")
    @test !contains(ir, "define %jl_value_t* @jlcall_")
    @test ismatch(r"define void @julia_foo_.+\(\) #0.+\{", ir)
end

@testset "exceptions" begin
    @eval codegen_exception() = throw(DivideError())
    ir = sprint(io->CUDAnative.code_llvm(io, codegen_exception, ()))

    # exceptions should get lowered to a plain trap...
    @test contains(ir, "llvm.trap")
    # not a jl_throw referencing a jl_value_t representing the exception
    @test !contains(ir, "jl_value_t")
    @test !contains(ir, "jl_throw")
end

@testset "sysimg" begin
    # bug: use a system image function

    @eval function codegen_call_sysimg(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
        return nothing
    end

    ir = sprint(io->CUDAnative.code_llvm(io, codegen_call_sysimg, (Ptr{Int},Int)))
    @test !contains(ir, "jlsys_")
end


############################################################################################

@testset "PTX assembly" begin

@testset "basic reflection" begin
    @eval ptx_valid_kernel(a) = (a[0] = 1; nothing)
    @eval ptx_invalid_kernel(a) = (a[0] = 1; 1)

    @test CUDAnative.code_ptx(DevNull, ptx_valid_kernel, Tuple{CuDeviceArray{Int,1}}) == nothing
    @test CUDAnative.code_ptx(DevNull, ptx_invalid_kernel, Tuple{CuDeviceArray{Int,1}}) == nothing
    @test_throws ErrorException CUDAnative.code_ptx(DevNull, ptx_invalid_kernel, Tuple{CuDeviceArray{Int,1}}; kernel=true) == nothing
end

@testset "entry-point functions" begin
    @eval @noinline function codegen_child(i)
        if i < 10
            return i*i
        else
            return (i-1)*(i+1)
        end
    end
    @eval codegen_parent(i) = (codegen_child(i); nothing)
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_parent, (Int64,); kernel=true))

    @test ismatch(r"\.visible \.entry julia_codegen_parent_", asm)
    @test ismatch(r"\.visible \.func .+ julia_codegen_child_", asm)
end

@testset "delayed lookup" begin
    @eval codegen_ref_nonexisting() = nonexisting
    @test_throws ErrorException CUDAnative.code_ptx(codegen_ref_nonexisting, ())
end

@testset "generic call" begin
    @eval codegen_call_nonexisting() = nonexisting()
    @test_throws ErrorException CUDAnative.code_ptx(codegen_call_nonexisting, ())
end


@testset "idempotency" begin
    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)

    @eval codegen_idempotency() = return nothing
    CUDAnative.code_ptx(DevNull, codegen_idempotency, ())
    CUDAnative.code_ptx(DevNull, codegen_idempotency, ())
end

@testset "child function reuse" begin
    # bug: depending on a child function from multiple parents resulted in
    #      the child only being present once

    @eval @noinline function codegen_child_reuse_child(i)
        if i < 10
            return i*i
        else
            return (i-1)*(i+1)
        end
    end

    @eval function codegen_child_reuse_parent1(arr::Ptr{Int64})
        i = codegen_child_reuse_child(0)
        unsafe_store!(arr, i, i)
        return nothing
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_parent1, (Ptr{Int64},)))
    @test ismatch(r".func .+ julia_codegen_child_reuse_child", asm)

    @eval function codegen_child_reuse_parent2(arr::Ptr{Int64})
        i = codegen_child_reuse_child(0)+1
        unsafe_store!(arr, i, i)

        return nothing
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_parent2, (Ptr{Int64},)))
    @test ismatch(r".func .+ julia_codegen_child_reuse_child", asm)
end

@testset "child function reuse bis" begin
    # bug: similar, but slightly different issue as above
    #      in the case of two child functions
    @eval @noinline function codegen_child_reuse_bis_child1()
        return 0
    end

    @eval @noinline function codegen_child_reuse_bis_child2()
        return 0
    end

    @eval function codegen_child_reuse_bis_parent1(arry::Ptr{Int64})
        i = codegen_child_reuse_bis_child1() + codegen_child_reuse_bis_child2()
        unsafe_store!(arry, i, i)

        return nothing
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_bis_parent1, (Ptr{Int64},)))

    @eval function codegen_child_reuse_bis_parent2(arry::Ptr{Int64})
        i = codegen_child_reuse_bis_child1() + codegen_child_reuse_bis_child2()
        unsafe_store!(arry, i, i+1)

        return nothing
    end
    asm = sprint(io->CUDAnative.code_ptx(io, codegen_child_reuse_bis_parent2, (Ptr{Int64},)))
end

@testset "nonsysimg recompilation" begin
    # issue #9: re-using non-sysimg functions should force recompilation
    #           (host fldmod1->mod1 throws)

    @eval function codegen_recompile(out)
        wid, lane = fldmod1(unsafe_load(out), Int32(32))
        unsafe_store!(out, wid)
        return nothing
    end

    asm = sprint(io->CUDAnative.code_ptx(io, codegen_recompile, (Ptr{Int32},)))
    @test !contains(asm, "jl_throw")
    @test !contains(asm, "jl_invoke")   # forced recompilation should still not invoke
end

@testset "reuse host sysimg" begin
    # issue #11: re-using host functions after PTX compilation
    @eval @noinline codegen_recompile_bis_child(x) = x+1

    @eval function codegen_recompile_bis_fromhost()
        codegen_recompile_bis_child(10)
    end

    @eval function codegen_recompile_bis_fromptx()
        codegen_recompile_bis_child(10)
        return nothing
    end

    CUDAnative.code_ptx(DevNull, codegen_recompile_bis_fromptx, ())
    CUDAnative.code_ptx(DevNull, codegen_recompile_bis_fromhost, ())
end

@testset "LLVM intrinsics" begin
    # issue #13 (a): cannot select trunc
    @eval codegen_issue_13(x) = convert(Int, x)
    CUDAnative.code_ptx(DevNull, codegen_issue_13, (Float64,))
end

end


############################################################################################

@testset "SASS" begin

@testset "basic reflection" begin
    @eval sass_valid_kernel(a) = (a[0] = 1; nothing)
    @eval sass_invalid_kernel(a) = (a[0] = 1; 1)

    @test CUDAnative.code_sass(DevNull, sass_valid_kernel, Tuple{CuDeviceArray{Int,1}}) == nothing
    @test_throws ErrorException CUDAnative.code_sass(DevNull, sass_invalid_kernel, Tuple{CuDeviceArray{Int,1}}) == nothing
end

end

############################################################################################

end

## LLVM IR

foo() = return nothing
ir = CUDAnative.code_llvm(foo, (); optimize=false, dump_module=true)

# module should contain our function + a generic call wrapper
@test contains(ir, "define void @julia_foo")
@test !contains(ir, "define %jl_value_t* @jlcall_")
@test ismatch(r"define void @julia_foo_.+\(\) #0.+\{", ir)


## PTX assembly

# TODO: PTX assembly generation / code_native
# -> test if foo and bar doesn't end up in same PTX module

# TODO: assert .entry
# TODO: assert devfun non .entry


function throw_exception()
    throw(DivideError())
end
ir = CUDAnative.code_llvm(throw_exception, ())

# exceptions should get lowered to a plain trap...
@test contains(ir, "llvm.trap")
# not a jl_throw referencing a jl_value_t representing the exception
@test !contains(ir, "jl_value_t")
@test !contains(ir, "jl_throw")

# delayed binding lookup (due to noexisting global)
ref_nonexisting() = nonexisting
@test_throws ErrorException CUDAnative.code_native(ref_nonexisting, ())

# generic call to nonexisting function
call_nonexisting() = nonexisting()
@test_throws ErrorException CUDAnative.code_native(call_nonexisting, ())

# bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)
codegen_twice() = return nothing
CUDAnative.code_native(codegen_twice, ())
CUDAnative.code_native(codegen_twice, ())

# bug: depending on a child function from multiple parents resulted in
#      the child only being present once
let
    @noinline function child(i)
        if i < 10
            return i*i
        else
            return (i-1)*(i+1)
        end
    end

    function parent1(arr::Ptr{Int64})
        i = child(0)
        unsafe_store!(arr, i, i)
        return nothing
    end
    asm = CUDAnative.code_native(parent1, (Ptr{Int64},))
    @test ismatch(r".func .+ julia_child", asm)

    function parent2(arr::Ptr{Int64})
        i = child(0)+1
        unsafe_store!(arr, i, i)

        return nothing
    end
    asm = CUDAnative.code_native(parent2, (Ptr{Int64},))
    @test ismatch(r".func .+ julia_child", asm)
end

# bug: similar, but slightly different issue as above
#      in the case of two child functions
let
    @noinline function child1()
        return 0
    end

    @noinline function child2()
        return 0
    end

    function parent1(arry::Ptr{Int64})
        i = child1() + child2()
        unsafe_store!(arry, i, i)

        return nothing
    end
    asm = CUDAnative.code_native(parent1, (Ptr{Int64},))


    function parent2(arry::Ptr{Int64})
        i = child1() + child2()
        unsafe_store!(arry, i, i+1)

        return nothing
    end
    asm = CUDAnative.code_native(parent2, (Ptr{Int64},))
end


# bug: use a system image function
let
    @noinline function call_sysimg(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
        return nothing
    end

    ccall(:jl_breakpoint, Void, (Any,), 42)
    ir = CUDAnative.code_llvm(call_sysimg, (Ptr{Int},Int))
    @test !contains(ir, "jlsys_")
end

# issue #9: re-using non-sysimg functions should force recompilation
#           (host fldmod1->mod1 throws)
let
    function kernel_9_b(out)
        wid, lane = fldmod1(unsafe_load(out), Int32(32))
        unsafe_store!(out, wid)
        return nothing
    end

    asm = CUDAnative.code_native(kernel_9_b, (Ptr{Int32},))
    @test !contains(asm, "jl_throw")
    @test !contains(asm, "jl_invoke")   # forced recompilation should still not invoke
end

# issue #11: re-using host functions after PTX compilation
let
    @noinline child_11(x) = x+1

    function kernel_11_host()
        child_11(10)
    end

    function kernel_11_ptx()
        child_11(10)
        return nothing
    end

    CUDAnative.code_native(kernel_11_ptx, ())
    CUDAnative.code_native(kernel_11_host, ())
end

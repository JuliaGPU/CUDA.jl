## LLVM IR

@target ptx foo() = return nothing
ir = CUDAnative.module_ir(foo, (); optimize=false)

# module should contain our function + a generic call wrapper
@test contains(ir, "define void @julia_foo")
@test contains(ir, "define %jl_value_t* @jlcall_foo")
@test ismatch(r"define void @julia_foo_.+\(\) #0.+\{", ir)
@test ismatch(r"define %jl_value_t\* @jlcall_", ir)
# code shouldn't contain a TLS pointer (PTX doesn't support TLS)
@test !contains(ir, "thread_ptr")


## PTX assembly

# TODO: PTX assembly generation / code_native
# -> test if foo and bar doesn't end up in same PTX module

# TODO: assert .entry
# TODO: assert devfun non .entry


@target ptx function throw_exception()
    throw(DivideError())
end
ir = CUDAnative.function_ir(throw_exception, ())

# exceptions should get lowered to a plain trap...
@test contains(ir, "llvm.trap")
# not a jl_throw referencing a jl_value_t representing the exception
@test !contains(ir, "jl_value_t")
@test !contains(ir, "jl_throw")

# delayed binding lookup (due to noexisting global)
@target ptx ref_nonexisting() = nonexisting
@test_throws ErrorException CUDAnative.module_asm(ref_nonexisting, ())

# generic call to nonexisting function
@target ptx call_nonexisting() = nonexisting()
@test_throws ErrorException CUDAnative.module_asm(call_nonexisting, ())

# cannot call PTX functions
@target ptx call_nonptx() = return nothing
@test_throws ErrorException call_nonptx()

# bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)
@target ptx codegen_twice() = return nothing
CUDAnative.module_asm(codegen_twice, ())
CUDAnative.module_asm(codegen_twice, ())

# bug: depending on a child function from multiple parents resulted in
#      the child only being present once
let
    @target ptx @noinline function child(i)
        if i < 10
            return i*i
        else
            return (i-1)*(i+1)
        end
    end

    @target ptx function parent1(arr::Ptr{Int64})
        i = child(0)
        unsafe_store!(arr, i, i)
        return nothing
    end
    asm = CUDAnative.module_asm(parent1, (Ptr{Int64},))
    @test ismatch(r".func .+ julia_child", asm)

    @target ptx function parent2(arr::Ptr{Int64})
        i = child(0)+1
        unsafe_store!(arr, i, i)

        return nothing
    end
    asm = CUDAnative.module_asm(parent2, (Ptr{Int64},))
    @test ismatch(r".func .+ julia_child", asm)
end

# bug: similar, but slightly different issue as above
#      in the case of two child functions
let
    @target ptx @noinline function child1()
        return 0
    end

    @target ptx @noinline function child2()
        return 0
    end

    @target ptx function parent1(arry::Ptr{Int64})
        i = child1() + child2()
        unsafe_store!(arry, i, i)

        return nothing
    end
    asm = CUDAnative.module_asm(parent1, (Ptr{Int64},))


    @target ptx function parent2(arry::Ptr{Int64})
        i = child1() + child2()
        unsafe_store!(arry, i, i+1)

        return nothing
    end
    asm = CUDAnative.module_asm(parent2, (Ptr{Int64},))
end


# bug: use a system image function
let
    @target ptx @noinline function call_sysimg(a,i)
        Base.pointerset(a, 0, mod1(i,10), 8)
        return nothing
    end

    ccall(:jl_breakpoint, Void, (Any,), 42)
    ir = CUDAnative.function_ir(call_sysimg, (Ptr{Int},Int))
    @test !contains(ir, "jlsys_")
end

# issue #9: not specifying '@target ptx' on child function should still work
let
    @noinline child_9_a() = throw(KeyError("whatever"))

    @target ptx function parent_9_a(out)
        if unsafe_load(out) == 0
            child_9_a()
        end
        return nothing
    end

    ir = CUDAnative.module_ir(parent_9_a, (Ptr{Int32},))
    @test contains(ir, "trap")
    @test !contains(ir, "jl_throw")
end

# issue #9: re-using non-sysimg functions should force recompilation
#           (host fldmod1->mod1 throws)
let
    @target ptx function kernel_9_b(out)
        wid, lane = fldmod1(unsafe_load(out), Int32(32))
        unsafe_store!(out, wid)
        return nothing
    end

    asm = CUDAnative.module_asm(kernel_9_b, (Ptr{Int32},))
    @test !contains(asm, "jl_throw")
    @test !contains(asm, "jl_invoke")   # forced recompilation should still not invoke
end

# issue #11: re-using host functions after PTX compilation
let
    @noinline child_11(x) = x+1

    function kernel_11_host()
        child_11(10)
    end

    @target ptx function kernel_11_ptx()
        child_11(10)
        return nothing
    end

    CUDAnative.module_asm(kernel_11_ptx, ())
    CUDAnative.module_asm(kernel_11_host, ())
end

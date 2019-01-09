# Dynamic Global Memory Allocation and Operations (B.21)

export malloc

@generated function malloc(sz::Csize_t)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))
    T_size = convert(LLVMType, Csize_t)
    T_ptr = convert(LLVMType, Ptr{Cvoid})

    # create function
    llvm_f, _ = create_function(T_ptr, [T_size])
    mod = LLVM.parent(llvm_f)

    # get the intrinsic
    # NOTE: LLVM doesn't have void*, Clang uses i8* for malloc too
    intr = LLVM.Function(mod, "malloc", LLVM.FunctionType(T_pint8, [T_size]))
    # should we attach some metadata here? julia.gc_alloc_obj has the following:
    #let attrs = function_attributes(f)
    #    AllocSizeNumElemsNotPresent = reinterpret(Cuint, Cint(-1))
    #    packed_allocsize = Int64(1) << 32 | AllocSizeNumElemsNotPresent
    #    push!(attrs, EnumAttribute("allocsize", packed_allocsize, ctx))
    #end
    #let attrs = return_attributes(f)
    #    push!(attrs, EnumAttribute("noalias", 0, ctx))
    #    push!(attrs, EnumAttribute("nonnull", 0, ctx))
    #end

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = call!(builder, intr, [parameters(llvm_f)[1]])

        jlptr = ptrtoint!(builder, ptr, T_ptr)

        ret!(builder, jlptr)
    end

    call_function(llvm_f, Ptr{Cvoid}, Tuple{Csize_t}, :((sz,)))
end

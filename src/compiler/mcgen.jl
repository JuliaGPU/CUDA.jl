# machine code generation

function machine(cap::VersionNumber, triple::String)
    InitializeNVPTXTarget()
    InitializeNVPTXTargetInfo()
    t = Target(triple)

    InitializeNVPTXTargetMC()
    cpu = "sm_$(cap.major)$(cap.minor)"
    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        # in the case of CUDA 9, we use sync intrinsics from PTX ISA 6.0+
        feat = "+ptx60"
    else
        feat = ""
    end
    tm = TargetMachine(t, triple, cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end

function mcgen(ctx::CompilerContext, mod::LLVM.Module, f::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

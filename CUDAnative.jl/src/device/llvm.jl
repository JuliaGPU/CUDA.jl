# wrappers for LLVM-specific functionality

@inline trap() = ccall("llvm.trap", llvmcall, Cvoid, ())

@inline assume(cond::Bool) =
    Base.llvmcall(("declare void @llvm.assume(i1)",
                    "%cond = icmp eq i8 %0, 1
                     call void @llvm.assume(i1 %cond)
                     ret void"), Nothing, Tuple{Bool}, cond)

# wrappers for LLVM-specific functionality

@inline trap() = ccall("llvm.trap", llvmcall, Cvoid, ())

@inline assume(cond::Bool) = Base.llvmcall(("""
        declare void @llvm.assume(i1)

        define void @entry(i8) #0 {
            %cond = icmp eq i8 %0, 1
            call void @llvm.assume(i1 %cond)
            ret void
        }

        attributes #0 = { alwaysinline }""", "entry"),
    Nothing, Tuple{Bool}, cond)

# do-block syntax
@inline function assume(f::F, val...) where {F}
    assume(f(val...))
    return val
end

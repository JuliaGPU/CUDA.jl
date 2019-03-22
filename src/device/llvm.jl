# wrappers for LLVM-specific functionality

@inline trap() = ccall("llvm.trap", llvmcall, Cvoid, ())

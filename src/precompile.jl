function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    @assert precompile(Tuple{CUDA.HostKernel{identity, Tuple{Nothing}},Nothing})
    @assert precompile(Tuple{Type{CuModule},String,Dict{CUDA.CUjit_option_enum, Any}})
    @assert precompile(Tuple{typeof(CUDA.initialize_cuda_context)})
    @assert precompile(Tuple{typeof(GPUCompiler.load_runtime),GPUCompiler.CompilerJob{GPUCompiler.PTXCompilerTarget, CUDA.CUDACompilerParams},LLVM.Context})
    @assert precompile(Tuple{typeof(cufunction),typeof(identity),Type{Tuple{Nothing}}})
    @assert precompile(Tuple{typeof(which(CUDA.pack_arguments,(Function,Vararg{Any, N} where N,)).generator.gen),Any,Any,Any})
end

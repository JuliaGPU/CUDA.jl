
# array
precompile(CuArray, (Vector{Int},))

# compilation
precompile(compiler_cache, (CuContext,))
#precompile(compiler_config, (CuDevice,))
precompile(compile, (CompilerJob,))
precompile(link, (CompilerJob,NamedTuple{(:image, :entry, :external_gvars), Tuple{Vector{UInt8}, String, Vector{String}}}))
precompile(create_exceptions!, (CuModule,))
precompile(run_and_collect, (Cmd,))

# launch
precompile(cudaconvert, (Function,))
precompile(Core.kwfunc(cudacall), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(cudacall),CuFunction,Type{Tuple{}}))
precompile(Core.kwfunc(launch), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(launch),CuFunction))

using PrecompileTools: @setup_workload, @compile_workload
@static if VERSION >= v"1.11.0-DEV.1603"
@setup_workload let
    @compile_workload begin
        target = PTXCompilerTarget(; cap=v"7.5")
        params = CUDACompilerParams(; cap=v"7.5", ptx=v"7.5")
        config = CompilerConfig(target, params)
        mi = GPUCompiler.methodinstance(typeof(identity), Tuple{Nothing})
        job = CompilerJob(mi, config)
        GPUCompiler.code_native(devnull, job)
    end
end
end

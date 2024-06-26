
# array
precompile(CuArray, (Vector{Int},))

# compilation
precompile(compiler_cache, (CuContext,))
#precompile(compiler_config, (CuDevice,))
precompile(compile, (CompilerJob,))
precompile(link, (CompilerJob,NamedTuple{(:image, :entry), Tuple{Vector{UInt8}, String}}))
precompile(create_exceptions!, (CuModule,))
precompile(run_and_collect, (Cmd,))

# launch
precompile(cudaconvert, (Function,))
precompile(Core.kwfunc(cudacall), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(cudacall),CuFunction,Type{Tuple{}}))
precompile(Core.kwfunc(launch), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(launch),CuFunction))

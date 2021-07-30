
# installation management
precompile(__init_toolkit__, ())
precompile(libcuda, ())

# array
precompile(CuArray, (Vector{Int},))

# compilation
precompile(CUDACompilerTarget, (CuDevice,))
precompile(cufunction_compile, (CompilerJob,))
precompile(cufunction_link, (CompilerJob,NamedTuple{(:image, :entry, :external_gvars), Tuple{Vector{UInt8}, String, Vector{String}}}))
precompile(cufunction_cache, (CuContext,))
precompile(create_exceptions!, (CuModule,))
precompile(run_and_collect, (Cmd,))

# launch
precompile(cudaconvert, (Function,))
precompile(Core.kwfunc(cudacall), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(cudacall),CuFunction,Type{Tuple{}}))
precompile(Core.kwfunc(launch), (NamedTuple{(:threads, :blocks), Tuple{Int64, Int64}},typeof(launch),CuFunction))

# compiler support for working with run-time libraries

function link_library!(job::CompilerJob, mod::LLVM.Module, lib::LLVM.Module)
    # linking is destructive, so copy the library
    lib = LLVM.Module(lib)

    # save list of external functions
    exports = String[]
    for f in functions(mod)
        fn = LLVM.name(f)
        if !haskey(functions(lib), fn)
            push!(exports, fn)
        end
    end

    link!(mod, lib)

    ModulePassManager() do pm
        # internalize all functions that aren't exports
        internalize!(pm, exports)

        # eliminate all unused internal functions
        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        run!(pm, mod)
    end
end

const libcache = Dict{String, LLVM.Module}()


#
# CUDA device library
#

function load_libdevice(cap)
    path = libdevice()

    get!(libcache, path) do
        open(path) do io
            parse(LLVM.Module, read(path), JuliaContext())
        end
    end
end

function link_libdevice!(job::CompilerJob, mod::LLVM.Module, lib::LLVM.Module)
    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    link_library!(job, mod, lib)

    ModulePassManager() do pm
        push!(metadata(mod), "nvvm-reflect-ftz",
              MDNode([ConstantInt(Int32(1), JuliaContext())]))
        run!(pm, mod)
    end
end


#
# CUDAnative run-time library
#

# get the path to a directory where we can put cache files (machine-specific, ephemeral)
# NOTE: maybe we should use XDG_CACHE_PATH/%LOCALAPPDATA%, but other Julia cache files
#       are put in .julia anyway so let's just follow suit for now.
function cachedir(depot=DEPOT_PATH[1])
    # this mimicks Base.compilecache. we can't just call the function, or we might actually
    # _generate_ a cache file, e.g., when running with `--compiled-modules=no`.
    if VERSION >= v"1.3.0-alpha.146"
        entrypath, entryfile = Base.cache_file_entry(Base.PkgId(CUDAnative))
        abspath(depot, entrypath, entryfile)
    else
        cachefile = abspath(depot, Base.cache_file_entry(Base.PkgId(CUDAnative)))

        # the cachefile consists of `/depot/compiled/vXXX/CUDAnative/$slug.ji`
        # transform that into `/depot/compiled/vXXX/CUDAnative/$slug/`
        splitext(cachefile)[1]
    end
end

# remove the existing cache globally, so any change to CUDAnative triggers recompilation.
rm(cachedir(); recursive=true, force=true)
# create an empty cache directory. since we only ever load from the first existing cachedir,
# this effectively invalidates preexisting caches in lower layers of the depot.
mkdir(cachedir())


## higher-level functionality to work with runtime functions

function LLVM.call!(builder, rt::Runtime.RuntimeMethodInstance, args=LLVM.Value[])
    bb = position(builder)
    f = LLVM.parent(bb)
    mod = LLVM.parent(f)

    # get or create a function prototype
    if haskey(functions(mod), rt.llvm_name)
        f = functions(mod)[rt.llvm_name]
        ft = eltype(llvmtype(f))
    else
        ft = LLVM.FunctionType(rt.llvm_return_type, rt.llvm_types)
        f = LLVM.Function(mod, rt.llvm_name, ft)
    end

    # runtime functions are written in Julia, while we're calling from LLVM,
    # this often results in argument type mismatches. try to fix some here.
    for (i,arg) in enumerate(args)
        if llvmtype(arg) != parameters(ft)[i]
            if (llvmtype(arg) isa LLVM.PointerType) &&
               (parameters(ft)[i] isa LLVM.IntegerType)
                # Julia pointers are passed as integers
                args[i] = ptrtoint!(builder, args[i], parameters(ft)[i])
            else
                error("Don't know how to convert ", arg, " argument to ", parameters(ft)[i])
            end
        end
    end

    call!(builder, f, args)
end


## functionality to build the runtime library

function emit_function!(mod, cap, f, types, name)
    tt = Base.to_tuple_type(types)
    new_mod, entry = codegen(:llvm, CompilerJob(f, tt, cap, #=kernel=# false);
                             libraries=false, strict=false)
    LLVM.name!(entry, name)
    link!(mod, new_mod)
end

function build_runtime(cap)
    mod = LLVM.Module("CUDAnative run-time library", JuliaContext())

    for method in values(Runtime.methods)
        emit_function!(mod, cap, method.def, method.types, method.llvm_name)
    end

    mod
end

function load_runtime(cap)
    # find the first existing cache directory (for when dealing with layered depots)
    cachedirs = [cachedir(depot) for depot in DEPOT_PATH]
    filter!(isdir, cachedirs)
    @assert !isempty(cachedirs)
    input_dir = first(cachedirs)

    # we are only guaranteed to be able to write in the current depot
    output_dir = cachedir()

    # if both aren't equal, copy pregenerated runtime libraries to our depot
    # NOTE: we don't just lazily read from the one and write to the other, because
    #       once we generate additional runtimes in the output dir we don't know if
    #       it's safe to load from other layers (since those could have been invalidated)
    if input_dir != output_dir
        mkpath(dirname(output_dir))
        cp(input_dir, output_dir)
    end

    name = "runtime_$(cap.major)$(cap.minor).bc"
    path = joinpath(output_dir, name)

    get!(libcache, path) do
        if ispath(path)
            open(path) do io
                parse(LLVM.Module, read(io), JuliaContext())
            end
        else
            @debug "Building the CUDAnative run-time library for sm_$(cap.major)$(cap.minor)."
            lib = build_runtime(cap)
            open(path, "w") do io
                write(io, lib)
            end
            lib
        end
    end
end

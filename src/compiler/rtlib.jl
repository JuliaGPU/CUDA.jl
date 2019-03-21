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

function find_libdevice(cap)
    CUDAnative.configured || return
    global libdevice

    if isa(libdevice, Dict)
        # select the most recent & compatible library
        vers = keys(CUDAnative.libdevice)
        compat_vers = Set(ver for ver in vers if ver <= cap)
        isempty(compat_vers) && error("No compatible CUDA device library available")
        ver = maximum(compat_vers)
        path = libdevice[ver]
    else
        libdevice
    end
end

function load_libdevice(cap)
    path = find_libdevice(cap)

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
        # TODO: run the reflect pass?
        run!(pm, mod)
    end
end


#
# CUDAnative run-time library
#

# remove existing runtime libraries globally,
# so any change to CUDAnative triggers recompilation
rm(joinpath(@__DIR__, "..", "..", "deps", "runtime"); recursive=true, force=true)


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
    new_mod, entry = compile(:llvm, cap, f, tt, #=kernel=# false; hooks=false)
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
    name = "cudanative.$(cap.major)$(cap.minor).bc"
    path = joinpath(@__DIR__, "..", "..", "deps", "runtime", name)
    mkpath(dirname(path))

    get!(libcache, path) do
        if ispath(path)
            open(path) do io
                parse(LLVM.Module, read(io), JuliaContext())
            end
        else
            @info "Building the CUDAnative run-time library for your sm_$(cap.major)$(cap.minor) device, this might take a while..."
            lib = build_runtime(cap)
            open(path, "w") do io
                write(io, lib)
            end
            lib
        end
    end
end

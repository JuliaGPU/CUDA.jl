# compiler support for working with run-time libraries

function link_library!(ctx::CompilerContext, mod::LLVM.Module, lib::LLVM.Module)
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

function link_libdevice!(ctx::CompilerContext, mod::LLVM.Module, lib::LLVM.Module)
    # override libdevice's triple and datalayout to avoid warnings
    triple!(lib, triple(mod))
    datalayout!(lib, datalayout(mod))

    link_library!(ctx, mod, lib)

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

function emit_function!(mod, cap, f, types, name)
    tt = Base.to_tuple_type(types)
    ctx = CompilerContext(f, tt, cap, #= kernel =# false)
    new_mod, entry = irgen(ctx)
    entry = optimize!(ctx, new_mod, entry)
    LLVM.name!(entry, name)

    link!(mod, new_mod)
end

function build_runtime(cap)
    @debug "Building CUDAnative run-time library for device capability $cap"
    mod = LLVM.Module("CUDAnative run-time library", JuliaContext())
    for binding in names(Runtime; all=true)
        value = getfield(Runtime, binding)
        if value isa Runtime.MethodInstance
            # TODO: check return type
            emit_function!(mod, cap, value.def, value.types, value.name)
        end
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
            lib = build_runtime(cap)
            open(path, "w") do io
                write(io, lib)
            end
            lib
        end
    end
end

function LLVM.call!(builder, rt::Runtime.MethodInstance, args=LLVM.Value[])
    bb = position(builder)
    f = LLVM.parent(bb)
    mod = LLVM.parent(f)

    # get or create a function prototype
    f = if haskey(functions(mod), rt.name)
        functions(mod)[rt.name]
    else
        ft = LLVM.FunctionType(rt.llvm_return_type, rt.llvm_types)
        f = LLVM.Function(mod, rt.name, ft)
    end

    call!(builder, f, args)
end

# remove existing runtime libraries globally,
# so any change to CUDAnative triggers recompilation
rm(joinpath(@__DIR__, "..", "..", "deps", "runtime"); recursive=true, force=true)

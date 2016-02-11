# Basic library loading and API calling

import Base: get

export
    @cucall

const libcuda = Ref{Ptr{Void}}()
const libcuda_vendor = Ref{ASCIIString}()
function load_library()
    try
        return (Libdl.dlopen(@windows ? "nvcuda.dll" : "libcuda"), "NVIDIA")
    end

    try
        return (Libdl.dlopen("libocelot"), "Ocelot")
    end

    error("Could not load CUDA (or any compatible) library")
end

function repr_indented(ex)
    # NOTE: this is clumsy, but otherwise the endlines get escaped
    io = IOBuffer()
    print(io, ex)
    str = takebuf_string(io)

    lines = split(strip(str), '\n')
    if length(lines) > 1
        for i = 1:length(lines)
            lines[i] = " "^7 * lines[i]
        end

        lines[1] = "\"\n" * lines[1]
        lines[length(lines)] = lines[length(lines)] * "\""

        return join(lines, '\n')
    else
        return str
    end
end

# API call wrapper
macro cucall(f, argtypes, args...)
    # Escape the tuple of arguments, making sure it is evaluated in caller scope
    # (there doesn't seem to be inline syntax like `$(esc(argtypes))` for this)
    esc_args = [esc(arg) for arg in args]

    blk = Expr(:block)

    if !isa(f, Expr) || f.head != :quote
        error("first argument to @cucall should be a symbol")
    end

    # Print the function name & arguments
    push!(blk.args, :(trace($(sprint(Base.show_unquoted,f.args[1])*"("); line=false)))
    i=length(args)
    for arg in args
        i-=1
        sep = (i>0 ? ", " : "")

        # TODO: we should only do this if evaluating `arg` has no side effects
        push!(blk.args, :(trace(repr_indented($(esc(arg))), $sep;
              prefix=$(sprint(Base.show_unquoted,arg))*"=", line=false)))
    end
    push!(blk.args, :(trace(""; prefix=") =", line=false)))

    # Generate the actual call and error check
    push!(blk.args, quote
        # NOTE: version symbol lookup needs to happen at runtime,
        #       because the mapping is created during __init__
        api_f = resolve($f)
        status = ccall(Libdl.dlsym(libcuda[], api_f), Cint,
                       $(esc(argtypes)), $(esc_args...))
        trace(CuError(status); prefix=" ")

        if status != SUCCESS.code
            err = CuError(status)
            throw(err)
        end
    end)

    blk
end

const api_mapping = Dict{Symbol,Symbol}()
resolve(f::Symbol) = get(api_mapping, f, f)

function __init_base__()
    (libcuda[], libcuda_vendor[]) = load_library()

    # Create mapping for versioned API calls
    if haskey(ENV, "CUDA_FORCE_API_VERSION")
        api_version = ENV["CUDA_FORCE_API_VERSION"]
    else
        api_version = driver_version()
    end
    populate_funmap(api_mapping, api_version)

    # Initialize the driver
    @cucall(:cuInit, (Cint,), 0)
end

function driver_version()
    version_ref = Ref{Cint}()
    @cucall(:cuDriverGetVersion, (Ptr{Cint},), version_ref)
    return version_ref[]
end

function vendor()
    return libcuda_vendor[]
end

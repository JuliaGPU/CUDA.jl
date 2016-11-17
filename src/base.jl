# Basic library loading and API calling

#
# API call wrapper
#

const libcuda = Ref{Ptr{Void}}()
function load_library()
    try
        return (Libdl.dlopen(@static is_windows() ? "nvcuda.dll" : "libcuda"), "NVIDIA")
    end

    error("Could not load CUDA (or any compatible) library")
end

macro apicall(f, argtypes, args...)
    # Escape the tuple of arguments, making sure it is evaluated in caller scope
    # (there doesn't seem to be inline syntax like `$(esc(argtypes))` for this)
    esc_args = [esc(arg) for arg in args]

    blk = Expr(:block)

    if !isa(f, Expr) || f.head != :quote
        error("first argument to @apicall should be a symbol")
    end

    # Print the function name & arguments
    if TRACE
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
    end

    # Generate the actual call
    @gensym status
    push!(blk.args, quote
        # NOTE: version symbol lookup needs to happen at runtime,
        #       because the mapping is created during __init__
        api_f = resolve($f)
        $status = ccall(Libdl.dlsym(libcuda[], api_f), Cint,
                       $(esc(argtypes)), $(esc_args...))
    end)

    # Print the results
    if TRACE
        push!(blk.args, :(trace(CuError($status); prefix=" ")))
    end

    # Check the return code
    push!(blk.args, quote
        if $status != SUCCESS.code
            err = CuError($status)
            throw(err)
        end
    end)

    return blk
end


#
# Initialization
#

const libcuda_vendor = Ref{String}()

function __init_library__()
    (libcuda[], libcuda_vendor[]) = load_library()
    __init_versioning__()
    init()
end

function init()
    @apicall(:cuInit, (Cint,), 0)
end

function vendor()
    return libcuda_vendor[]
end

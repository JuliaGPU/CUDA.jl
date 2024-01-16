# code reflection entry-points

using .CUPTI
using .CUPTI: CUpti_ModuleResourceData



#
# code_* replacements
#

"""
    code_sass([io], f, types; raw=false)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `stdout`.

The following keyword arguments are supported:

- `raw`: dump the assembly like `nvdisasm` reports it, without post-processing;
- all keyword arguments from [`cufunction`](@ref)

See also: [`@device_code_sass`](@ref)
"""
function code_sass(io::IO, @nospecialize(func), @nospecialize(types); kwargs...)
    compiler_kwargs, kwargs = split_kwargs_runtime(kwargs, COMPILER_KWARGS)
    source = methodinstance(typeof(func), Base.to_tuple_type(types))
    config = compiler_config(device(); compiler_kwargs...)
    job = CompilerJob(source, config)
    code_sass(io, job; kwargs...)
end

code_sass(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_sass(stdout, func, types; kwargs...)

function code_sass(io::IO, job::CompilerJob; raw::Bool=false)
    if !job.config.kernel
        error("Can only generate SASS code for kernel functions")
    end

    # NVIDIA bug #3964667: CUPTI in CUDA 11.7+ broken for sm_35 devices
    if runtime_version() >= v"11.7" && capability(device()) <= v"3.7"
        @error """SASS code generation is not supported on this device.
                  Please downgrade to CUDA 11.6 or lower, or use a more recent device."""
        return
    end

    cfg = CUPTI.CallbackConfig([CUPTI.CUPTI_CB_DOMAIN_RESOURCE]) do domain, id, data
        id == CUPTI.CUPTI_CBID_RESOURCE_MODULE_LOADED || return
        resourceDescriptor =
            unsafe_load(convert(Ptr{CUpti_ModuleResourceData}, data.resourceDescriptor))
        cubin = unsafe_wrap(Vector{Cchar}, pointer(resourceDescriptor.pCubin),
                            resourceDescriptor.cubinSize)
        disassemble_cubin(io, cubin; raw)
    end

    compiled = compile(job)
    CUPTI.enable!(cfg)
    try
        link(job, compiled)
    finally
        CUPTI.disable!(cfg)
    end
  
    return
end

# disassemble a cubin to SASS
function disassemble_cubin(io::IO, cubin::Vector{Cchar}; raw::Bool)
    mktemp() do cubin_path,cubin_io
        write(cubin_io, cubin)
        flush(cubin_io)

        cmd = `$(nvdisasm()) --print-code --print-line-info $cubin_path`
        for line in readlines(cmd)
            if !raw
                # nvdisasm output is pretty verbose;
                # perform some clean-up and make it look like @code_native
                line = replace(line, r"/\*[0-9a-f]{4}\*/" => "        ") # strip inst addr
                line = replace(line, r"^[ ]{30}" => "   ")               # reduce leading spaces
                line = replace(line, r"[\s+]//##" => ";")                # change line info tag
                line = replace(line, r"^\." => "\n.")                    # break before new BBs
                line = replace(line, r"; File \"(.+?)\", line (\d+)" => s"; Location \1:\2") # rename line info
            end
            println(io, line)
        end
    end
end


# forward the rest to GPUCompiler with an appropriate CompilerJob

# function to split off certain kwargs for selective forwarding, at run time.
# `@cuda` does something similar at parse time, using `GPUCompiler.split_kwargs`.
function split_kwargs_runtime(kwargs, wanted::Vector{Symbol})
    remaining = Dict{Symbol, Any}()
    extracted = Dict{Symbol, Any}()
    for (key, value) in kwargs
        if key in wanted
            extracted[key] = value
        else
            remaining[key] = value
        end
    end
    return extracted, remaining
end

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)

    @eval begin
        function $method(io::IO, @nospecialize(func), @nospecialize(types);
                         kernel=false, kwargs...)
            compiler_kwargs, kwargs = split_kwargs_runtime(kwargs, COMPILER_KWARGS)
            source = methodinstance(typeof(func), Base.to_tuple_type(types))
            config = compiler_config(device(); kernel, compiler_kwargs...)
            job = CompilerJob(source, config)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

const code_ptx = code_native

"""
    CUDA.return_type(f, tt) -> r::Type

Return a type `r` such that `f(args...)::r` where `args::tt`.
"""
function return_type(@nospecialize(func), @nospecialize(tt))
    source = methodinstance(typeof(func), tt)
    config = compiler_config(device())
    job = CompilerJob(source, config)
    interp = GPUCompiler.get_interpreter(job)
    sig = Base.signature_type(func, tt)
    Core.Compiler.return_type(interp, sig)
end


#
# @device_code_* functions
#

export @device_code_lowered, @device_code_typed, @device_code_warntype,
       @device_code_llvm, @device_code_ptx, @device_code_sass,
       @device_code

"""
    @device_code_sass [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`CUDA.code_sass`](@ref) to
`io` for every compiled CUDA kernel. For other supported keywords, see
[`CUDA.code_sass`](@ref).
"""
macro device_code_sass(ex...)
    function hook(job::CompilerJob; io::IO=stdout, kwargs...)
        println(io, "// $job")
        println(io)
        code_sass(io, job; kwargs...)
    end
    GPUCompiler.emit_hooked_compilation(hook, ex...)
end


# forward the rest to GPUCompiler
@eval $(Symbol("@device_code_lowered")) = $(getfield(GPUCompiler, Symbol("@device_code_lowered")))
@eval $(Symbol("@device_code_typed")) = $(getfield(GPUCompiler, Symbol("@device_code_typed")))
@eval $(Symbol("@device_code_warntype")) = $(getfield(GPUCompiler, Symbol("@device_code_warntype")))
@eval $(Symbol("@device_code_llvm")) = $(getfield(GPUCompiler, Symbol("@device_code_llvm")))
@eval $(Symbol("@device_code_ptx")) = $(getfield(GPUCompiler, Symbol("@device_code_native")))
@eval $(Symbol("@device_code")) = $(getfield(GPUCompiler, Symbol("@device_code")))

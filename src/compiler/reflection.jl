# code reflection entry-points

using .CUPTI: CUpti_CallbackDomain, CUpti_CallbackId, CUpti_SubscriberHandle,
              CUpti_ResourceData, CUpti_ModuleResourceData



#
# code_* replacements
#

function code_sass_callback(userdata::Ptr{Cvoid}, domain::CUpti_CallbackDomain,
                            cbid::CUpti_CallbackId, cbdada::Ptr{Cvoid})
    dest = Base.unsafe_pointer_to_objref(userdata)::Ref{Any}

    if domain == CUPTI.CUPTI_CB_DOMAIN_RESOURCE
        cbdada = unsafe_load(reinterpret(Ptr{CUpti_ResourceData}, cbdada))
        if cbid == CUPTI.CUPTI_CBID_RESOURCE_MODULE_LOADED
            resourceDescriptor =
                unsafe_load(reinterpret(Ptr{CUpti_ModuleResourceData}, cbdada.resourceDescriptor))
            cubin = unsafe_wrap(Vector{Cchar}, pointer(resourceDescriptor.pCubin),
                                resourceDescriptor.cubinSize)
            dest[] = copy(cubin)
        end
    end

    return
end

"""
    code_sass([io], f, types, cap::VersionNumber)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `stdout`.

The following keyword arguments are supported:

- `cap` which device to generate code for
- `kernel`: treat the function as an entry-point kernel
- `verbose`: enable verbose mode, which displays code generation statistics

See also: [`@device_code_sass`](@ref)
"""
function code_sass(io::IO, @nospecialize(func), @nospecialize(types), kernel::Bool=true;
                   verbose::Bool=false, kwargs...)
    tt = Base.to_tuple_type(types)
    target = PTXCompilerTarget(; cap=supported_capability(device()), kwargs...)
    params = CUDACompilerParams()
    job = CompilerJob(target, FunctionSpec(func, tt, kernel), params)
    code_sass(io, job; verbose=verbose)
end

function code_sass(io::IO, job::CUDACompilerJob; verbose::Bool=false)
    if !job.source.kernel
        error("Can only generate SASS code for kernel functions")
    end

    asm, _ = GPUCompiler.codegen(:asm, job)

    cubin = Ref{Any}()
    callback = @cfunction(code_sass_callback, Cvoid,
                          (Ptr{Cvoid}, CUpti_CallbackDomain, CUpti_CallbackId, Ptr{Cvoid}))

    # JIT compile and capture the generated object file
    subscriber_ref = Ref{CUpti_SubscriberHandle}()
    res = CUPTI.unsafe_cuptiSubscribe(subscriber_ref, callback, Base.pointer_from_objref(cubin))
    if res === CUPTI.CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
        error("""Insufficient priviliges: You don't have permissions to profile GPU code, which is required for `code_sass`.
                 Get administrative priviles or allow all users to profile: https://developer.nvidia.com/ERR_NVGPUCTRPERM#SolnAdminTag""")
    elseif res != CUPTI.CUPTI_SUCCESS
        throw(CUPTIError(res))
    end
    subscriber = subscriber_ref[]
    try
        CUPTI.cuptiEnableDomain(1, subscriber, CUPTI.CUPTI_CB_DOMAIN_RESOURCE)
        CuModule(asm)
    finally
        CUPTI.cuptiUnsubscribe(subscriber)
    end

    # disassemble to SASS
    isassigned(cubin) || error("No kernels compiled")
    mktemp() do cubin_path,cubin_io
        write(cubin_io, cubin[])
        flush(cubin_io)

        cmd = `$(nvdisasm()) --print-code --print-line-info $cubin_path`
        for line in readlines(cmd)
            # nvdisasm output is pretty verbose;
            # perform some clean-up and make it look like @code_native
            line = replace(line, r"/\*[0-9a-f]{4}\*/" => "        ") # strip inst addr
            line = replace(line, r"^[ ]{30}" => "   ")               # reduce leading spaces
            line = replace(line, r"[\s+]//##" => ";")                # change line info tag
            line = replace(line, r"^\." => "\n.")                    # break before new BBs
            line = replace(line, r"; File \"(.+?)\", line (\d+)" => s"; Location \1:\2") # rename line info
            println(io, line)
        end
    end
end

code_sass(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_sass(stdout, func, types; kwargs...)


# forward the rest to GPUCompiler with an appropriate CompilerJob

for method in (:code_typed, :code_warntype, :code_llvm, :code_native)
    # only code_typed doesn't take a io argument
    args = method == :code_typed ? (:job,) : (:io, :job)

    @eval begin
        function $method(io::IO, @nospecialize(func), @nospecialize(types);
                         kernel::Bool=false, minthreads=nothing, maxthreads=nothing,
                         blocks_per_sm=nothing, maxregs=nothing, kwargs...)
            source = FunctionSpec(func, Base.to_tuple_type(types), kernel)
            target = PTXCompilerTarget(; cap=supported_capability(device()),
                                       minthreads=minthreads, maxthreads=maxthreads,
                                       blocks_per_sm=blocks_per_sm, maxregs=maxregs)
            params = CUDACompilerParams()
            job = CompilerJob(target, source, params)
            GPUCompiler.$method($(args...); kwargs...)
        end
        $method(@nospecialize(func), @nospecialize(types); kwargs...) =
            $method(stdout, func, types; kwargs...)
    end
end

const code_ptx = code_native



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
    function hook(job::CUDACompilerJob; io::IO=stdout, kwargs...)
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

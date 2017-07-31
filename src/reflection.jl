# code reflection entry-points

export
    @code_lowered, @code_typed, @code_warntype,
    code_llvm, code_ptx, code_sass, @code_llvm, @code_ptx, @code_sass

#
# code_* replacements
#

# Return the capability of the current context's device, or a sane fall-back.
function current_capability()
    fallback = v"2.0"
    if !isdefined(CUDAdrv, :configured) || !CUDAdrv.configured
        return fallback
    end

    ctx = CuCurrentContext()
    if isnull(ctx)
        return fallback
    end

    return capability(device(ctx))
end

"""
    code_llvm([io], f, types; optimize=true, dump_module=false, cap::VersionNumber)

Prints the LLVM IR generated for the method matching the given generic function and type
signature to `io` which defaults to `STDOUT`. The IR is optimized according to `optimize`
(defaults to true), and the entire module, including headers and other functions, is dumped
if `dump_module` is set (defaults to false). The device capability `cap` to generate code
for defaults to the current active device's capability, or v"2.0" if there is no such active
context.
"""
function code_llvm(io::IO, func::ANY, types::ANY=Tuple;
                   optimize::Bool=true, dump_module::Bool=false,
                   cap::VersionNumber=current_capability(), kernel::Bool=false)
    tt = Base.to_tuple_type(types)
    mod, entry = irgen(func, tt; kernel=kernel)
    if optimize
        optimize!(mod, cap)
    end
    if dump_module
        show(io, mod)
    else
        show(io, entry)
    end
end
code_llvm(func::ANY, types::ANY=Tuple; kwargs...) = code_llvm(STDOUT, func, types; kwargs...)

"""
    code_ptx([io], f, types; cap::VersionNumber, kernel::Bool=false)

Prints the PTX assembly generated for the method matching the given generic function and
type signature to `io` which defaults to `STDOUT`. The device capability `cap` to generate
code for defaults to the current active device's capability, or v"2.0" if there is no such
active context. The optional `kernel` parameter indicates whether the function in question
is an entry-point function, or a regular device function.
"""
function code_ptx(io::IO, func::ANY, types::ANY=Tuple;
                  cap::VersionNumber=current_capability(), kernel::Bool=false)
    @assert isa(func, Core.Function)
    tt = Base.to_tuple_type(types)
    kernel && check_kernel(func, tt)

    ptx,_ = compile_function(func, tt, cap; kernel=kernel)
    # TODO: this code contains all the functions in the call chain,
    #       is it possible to implement `dump_module`?
    print(io, ptx)
end
code_ptx(func::ANY, types::ANY=Tuple; kwargs...) = code_ptx(STDOUT, func, types; kwargs...)

"""
    code_sass([io], f, types, cap::VersionNumber)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `STDOUT`. The device capability `cap` to generate code
for defaults to the current active device's capability, or v"2.0" if there is no such active
context.

Note that the method needs to be a valid entry-point kernel, ie. it should not return any
values.
"""
function code_sass(io::IO, func::ANY, types::ANY=Tuple;
                   cap::VersionNumber=current_capability())
    @assert isa(func, Core.Function)
    tt = Base.to_tuple_type(types)
    check_kernel(func, tt)

    ptx,_ = compile_function(func, tt, cap)

    fn = tempname()
    gpu = "sm_$(cap.major)$(cap.minor)"
    # NOTE: this might not match what is being executed, due to the PTX->SASS conversion
    #       by the driver possibly not matching what `ptxas` (part of the toolkit) does.
    # TODO: see how `nvvp` extracts SASS code when doing PC sampling, and copy that.
    Base.run(`ptxas --gpu-name $gpu --output-file $fn --input-as-string $ptx`)
    try
        print(io, readstring(`cuobjdump --dump-sass $fn`))
    finally
        rm(fn)
    end
end
code_sass(func::ANY, types::ANY=Tuple; kwargs...) = code_sass(STDOUT, func, types; kwargs...)


#
# @code_* replacements
#

function gen_call_with_extracted_types(f, ex)
    :($f($(esc(ex.args[1])), Base.typesof(cudaconvert.(($(esc.(ex.args[2:end])...),))...)))
end

for (fname,kernel_arg) in [(:code_lowered, false), (:code_typed, false), (:code_warntype, false),
                           (:code_llvm, true), (:code_ptx, true), (:code_sass, false)]
    # TODO: test the kernel_arg-based behavior
    @eval begin
        @doc $"""
            $fname
        Extracts the relevant function call from any `@cuda` invocation, evaluates the
        arguments to the function or macro call, determines their types (taking into account
        GPU-specific type conversions), and calls $fname on the resulting expression.
        Can be applied to a pure function call, or a call prefixed with the `@cuda` macro.
        In that case, kernel code generation conventions are used (wrt. argument conversions,
        return values, etc).
        """ macro $(fname)(ex0)
            if ex0.head == :macrocall
                # @cuda (...) f()
                if Base.VERSION >= v"0.7.0-DEV.357"
                    ex0 = ex0.args[4]
                else
                    ex0 = ex0.args[3]
                end
                kernel = true
            else
                kernel = false
            end

            wrapper(func, types) = $kernel_arg ? $fname(func, types, kernel = kernel) :
                                                 $fname(func, types)

            gen_call_with_extracted_types(wrapper, ex0)
        end
    end
end

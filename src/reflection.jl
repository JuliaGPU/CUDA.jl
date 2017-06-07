# code reflection entry-points

export
    @code_lowered, @code_typed, @code_warntype,
    code_llvm, code_ptx, code_sass, @code_llvm, @code_ptx, @code_sass

#
# code_* replacements
#

# Return the capability of the current context's device, or a sane fall-back.
function current_capability()
    ctx = CuCurrentContext()
    if isnull(ctx)
        return v"2.0"
    else
        return capability(device(ctx))
    end
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
                   cap::VersionNumber=current_capability())
    mod, entry = irgen(func, types)
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

for fname in [:code_lowered, :code_typed, :code_warntype, :code_llvm, :code_ptx, :code_sass]
    # types often need to be converted (eg. CuArray -> CuDeviceArray),
    # so generate a type-converting wrapper, and a macro to call it
    fname_wrapper = Symbol(fname, :_cputyped)

    @eval begin
        function $fname_wrapper(func, types)
            _, codegen_types, _ =
                convert_arguments(fill(Symbol(), length(types.parameters)),
                                  types.parameters)
            $fname(func, codegen_types)
        end

        @doc $"""
            $fname

        Extracts the relevant function call from any `@cuda` invocation, evaluates the
        arguments to the function or macro call, determines their types (taking into account
        GPU-specific type conversions), and calls $fname on the resulting expression.
        """ macro $(fname)(ex0)
            if ex0.head == :macrocall
                # @cuda (...) f()
                ex0 = ex0.args[3]
            end

            if Base.VERSION >= v"0.7.0-DEV.481"
                Base.gen_call_with_extracted_types(__module__, $(Expr(:quote,fname_wrapper)), ex0)
            else
                Base.gen_call_with_extracted_types($(Expr(:quote,fname_wrapper)), ex0)
            end
        end
    end
end

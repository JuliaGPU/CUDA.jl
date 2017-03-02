# code reflection entry-points


#
# code_* replacements
#

# NOTE: default capability is a sane one for testing purposes

function code_llvm(io::IO, func::ANY, types::ANY=Tuple;
                   optimize::Bool=true, dump_module::Bool=false, cap::VersionNumber=v"2.0")
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
    code_ptx([io], f, types, cap=v"2.0")

Prints the PTX assembly generated for the method matching the given generic function and
type signature to `io` which defaults to `STDOUT`.
"""
function code_ptx(io::IO, func::ANY, types::ANY=Tuple;
                  cap::VersionNumber=v"2.0", kernel::Bool=false)
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
    code_sass([io], f, types, cap=v"2.0")

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `STDOUT`.

Note that the method needs to be a valid entry-point kernel, ie. it should not return any
values.
"""
function code_sass(io::IO, func::ANY, types::ANY=Tuple; cap::VersionNumber=v"2.0")
    @assert isa(func, Core.Function)
    tt = Base.to_tuple_type(types)
    check_kernel(func, tt)

    ptx,_ = compile_function(func, tt, cap)

    fn = tempname()
    gpu = "sm_$(cap.major)$(cap.minor)"
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

for fname in [:code_llvm, :code_ptx, :code_sass]
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

        macro ($fname)(ex0)
            if ex0.head == :macrocall
                # @cuda (...) f()
                ex0 = ex0.args[3]
            end
            Base.gen_call_with_extracted_types($(Expr(:quote,fname_wrapper)), ex0)
        end
    end
end

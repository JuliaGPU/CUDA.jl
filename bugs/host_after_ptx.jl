#=
This doesn't work because functions are only compiled once, disregarding @target.

More specifically, jl_compile_linfo only compiles if functionObject is not set.
If the function has already been compiled as part of another cycle, its module
has already been finalized, and in turn consumed (ie. removed from
module_for_fname) by finalize_function.

This means that later uses will not trigger a new compilation because the
functionObject is already set, and consequently no module will be finalized
which means no entry in module_for_fname. Consequently, finalization of the
parent function will fail because the required module cannot be found.
=#

@noinline child(x) = x+1

function f_host()
    child(10)
end

@target ptx function f_ptx()
    child(10)
    return nothing
end

function main()
    code_native(f_ptx, ())
    code_native(f_host, ())
end

main()

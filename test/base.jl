@testset "base interface" begin

############################################################################################

@testset "method caching" begin

import InteractiveUtils: _dump_function

# #17057 fallout
@eval @noinline post17057_child(i) = sink(i)
@eval function post17057_parent(arr::Ptr{Int64})
    i = post17057_child(0)
    unsafe_store!(arr, i, i)
    return nothing
end

# bug 1: emit_invoke performed dynamic call due to NULL child function
#        (hooked module activation because of bug 2 below)
if VERSION < v"0.7.0-DEV.1669"
    hook_module_activation(ref::Ptr{Cvoid}) = nothing
    hooks = Base.CodegenHooks(module_activation=hook_module_activation)
    params = Base.CodegenParams(cached=false, runtime=false, hooks=hooks)
    _dump_function(post17057_parent, Tuple{Ptr{Int64}},
                   #=native=#false, #=wrapper=#false, #=strip=#false,
                   #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
                   params)
end

# bug 2: default module activation segfaulted on NULL child function if cached=false
params = Base.CodegenParams(cached=false)
_dump_function(post17057_parent, Tuple{Ptr{Int64}},
               #=native=#false, #=wrapper=#false, #=strip=#false,
               #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
               params)

end

############################################################################################

end
@testset "base interface" begin

############################################################################################

@testset "method caching" begin

# #17057 fallout
@eval @noinline function post17057_child(i)
    if i < 10
        return i*i
    else
        return (i-1)*(i+1)
    end
end

@eval function post17057_parent(arr::Ptr{Int64})
    i = post17057_child(0)
    unsafe_store!(arr, i, i)
    return nothing
end

# bug 1: emit_invoke performed dynamic call due to NULL child function
#        (hooked module activation because of bug 2 below)
hook_module_activation(ref::Ptr{Void}) = nothing
hooks = Base.CodegenHooks(module_activation=hook_module_activation)
params = Base.CodegenParams(cached=false, runtime=false, hooks=hooks)
Base._dump_function(post17057_parent, Tuple{Ptr{Int64}},
                    #=native=#false, #=wrapper=#false, #=strip=#false,
                    #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
                    params)

# bug 2: default module activation segfaulted on NULL child function if cached=false
params = Base.CodegenParams(cached=false)
Base._dump_function(post17057_parent, Tuple{Ptr{Int64}},
                    #=native=#false, #=wrapper=#false, #=strip=#false,
                    #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
                    params)

end

############################################################################################

end
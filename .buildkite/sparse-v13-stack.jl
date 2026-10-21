# TEMPORARY coordinated-v13 stack validation. Remove after registered resolution works.
using TOML
const stack_backend_root=normpath(joinpath(@__DIR__,".."))
const stack_backend_name=TOML.parsefile(joinpath(stack_backend_root,"Project.toml"))["name"]
const stack_gpuarrays_revision="dd0ee48421b1a3b713ee88a90e7f7274c5a39b6d"
const stack_arguments=isempty(ARGS) ? Base.shell_split(get(ENV,"GPUARRAYS_V13_STACK_TEST_ARGS","")) : copy(ARGS)
const stack_keep_sources=any(arg->startswith(arg,"--setup-project=") || arg=="--setup-only",stack_arguments)
const stack_checkout=mktempdir(;prefix=lowercase(stack_backend_name)*"-sparse-v13-",cleanup=!stack_keep_sources)
run(`git clone --quiet --filter=blob:none https://github.com/JuliaGPU/GPUArrays.jl $stack_checkout`)
run(`git -C $stack_checkout checkout --quiet --detach $stack_gpuarrays_revision`)
readchomp(`git -C $stack_checkout rev-parse HEAD`)==stack_gpuarrays_revision || error("wrong GPUArrays revision")
ENV["GPUARRAYS_V13_STACK_BACKEND"]=stack_backend_root

function merge_stack_preferences!(destination,source)
    # Match Base's recursive preference merge, including explicit deletions.
    cleared=get(source,"__clear__",nothing)
    if cleared isa Vector{String}
        for key in cleared
            delete!(destination,key)
        end
    end
    for (key,value) in source
        if value isa AbstractDict && get(destination,key,nothing) isa AbstractDict
            merge_stack_preferences!(destination[key],value)
        else
            destination[key]=deepcopy(value)
        end
    end
end
const stack_preferences=Dict{String,Any}()
if haskey(ENV,"GPUARRAYS_V13_STACK_PREFERENCES")
    isfile(ENV["GPUARRAYS_V13_STACK_PREFERENCES"]) || error("missing supplied preferences")
end
for path in (joinpath(stack_backend_root,"LocalPreferences.toml"),
             joinpath(stack_backend_root,"test","LocalPreferences.toml"),
             get(ENV,"GPUARRAYS_V13_STACK_PREFERENCES",""))
    isempty(path) || !isfile(path) || merge_stack_preferences!(stack_preferences,TOML.parsefile(path))
end
if get(ENV,"GPUARRAYS_V13_STACK_ONEAPI_DEBUG","false")=="true"
    merge_stack_preferences!(stack_preferences,Dict(
        "NEO_jll"=>Dict("debug"=>"true"),"libigc_jll"=>Dict("debug"=>"true")))
end
if !isempty(stack_preferences)
    directory=mktempdir(;prefix="sparse-v13-preferences-",cleanup=!stack_keep_sources)
    path=joinpath(directory,"LocalPreferences.toml")
    open(path,"w") do io
        TOML.print(io,stack_preferences)
    end
    ENV["GPUARRAYS_V13_STACK_PREFERENCES"]=path
end

const stack_setup_arg=findfirst(arg->startswith(arg,"--setup-project="),stack_arguments)
if stack_setup_arg!==nothing
    target=split(stack_arguments[stack_setup_arg],'=';limit=2)[2]
    target in ("docs","perf","examples",".buildkite/stack-env") || error("unsupported setup project")
    target==".buildkite/stack-env" || isfile(joinpath(stack_backend_root,target,"Project.toml")) ||
        error("missing setup project: $target")
    ENV["GPUARRAYS_V13_STACK_SETUP_PROJECT"]=joinpath(stack_backend_root,target)
    deleteat!(stack_arguments,stack_setup_arg)
    push!(stack_arguments,"--setup-only")
end
empty!(ARGS)
append!(ARGS,[stack_backend_name;stack_arguments])
include(joinpath(stack_checkout,".buildkite","sparse-v13-stack.jl"))

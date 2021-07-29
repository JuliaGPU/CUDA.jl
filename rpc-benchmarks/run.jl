if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg

    if isfile("Project.toml")
        Pkg.activate("./")
    end
end

using CUDA
using Statistics
using BenchmarkTools
using LLVM
using LLVM.Interop
using Core: LLVMPtr

abstract type BenchRunner end

# Get identifying name
name(::BenchRunner) = ""
# Execute something to warm up
warmup(::BenchRunner) = ()
# Actually execute the runner for 'threads' and 'blocks', returning the execution times
execute(::BenchRunner, threads::Int, blocks::Int; samples=5, evals=1) = []
filter(::BenchRunner, _threads, _blocks) = true

struct SimpleRunner <: BenchRunner
    name::String
    fie::Function
    filter::Function
    setup::Expr
end

SimpleRunner(name, fie, setup::Expr, filter=((_, _) -> true)) = SimpleRunner(name, fie, filter, setup)
SimpleRunner(name, fie, filter=((_, _) -> true)) = (SimpleRunner(name, fie, filter, quote ((_, _) -> []) end))
name(runner::SimpleRunner) = runner.name
warmup(runner::SimpleRunner) = eval(runner.fie(1,1))
filter(runner::SimpleRunner, threads, blocks) = runner.filter(threads, blocks)
function execute(runner::SimpleRunner, threads::Int, blocks::Int;samples=5, evals=1)
    quoted = runner.fie(threads, blocks)
    trial = @benchmark ($quoted(args...)) samples=samples evals=evals setup=(args = $(runner.setup)($threads, $blocks))
    return trial.times
end


struct ExecRunner <:BenchRunner
    name::String
    exec_name::String
end

name(runner::ExecRunner) = runner.name
warmup(::ExecRunner) = ()
function execute(runner::ExecRunner, threads::Int, blocks::Int;samples=5, evals=1)
    evals == 1 || error("Multiple evals per sample is not supported")

    manager = CUDA.SimpleAreaManager(div(threads * blocks, 32, RoundUp), 100)

    times = []
    for _ in 1:samples
        # exec command
        t = exec_cmd(runner.exec_name, threads, blocks, manager; target="seconds") * 1000 * 1000 * 1000 # ns to seconds
        push!(times, t)
    end

    return times
end



include("util.jl")
include("des_bench.jl")
include("malloc_bench.jl")
include("filter_bench.jl")
include("print_bench.jl")

function perform_benchmarks(threads, blocks, runners; io = stdout, samples=5, evals=1, measures=[:all])
    print(io, "\"threads\",\"blocks\"")
    for runner in runners
        print(io, ",$(trial_header(name(runner), measures))")
    end
    println(io, "")

    for thread_count in threads
        for block_count in blocks
            print(io, "$thread_count,$block_count")

            for runner in runners
                println(name(runner))
                times = filter(runner, thread_count, block_count) ? execute(runner,thread_count,block_count; samples=samples, evals=evals) : [0,0,0]
                print(io, ",$(times_to_string(times, measures))")
                flush(io)
            end

            println(io, "")
            flush(io)
        end
    end
end


function exec_cmd(args...;target="seconds")
    out = IOBuffer()

    r = get_exec_cmd(args...)
    run(pipeline(r, stdout=out))
    rr = String(take!(out))
    words = split(rr)
    ix = findfirst(x -> x == target, words)
    time = parse(Float32, words[ix-1])
    return time
end


function get_exec_cmd(args...)
    cmd = push!(collect(Base.julia_cmd()), @__FILE__,ARGS...)

    ix = findfirst(x -> x == "run", cmd)
    ix === nothing && error("Could not change command!")

    cmd[ix] = "exec"
    cmd = cmd[1:ix]

    push!(cmd, [string(x) for x in args]...)
    return Cmd(Cmd(cmd); ignorestatus=true)
end


if abspath(PROGRAM_FILE) == @__FILE__
    cmds = ["run", "exec"]

    cmd = push!(collect(Base.julia_cmd()), @__FILE__,ARGS...)

    ix = findfirst(x -> in(x, cmds), cmd)
    ix === nothing && (error("Could not find command!"); exit(1))

    type = cmd[ix]

    if type === "run"
        arg = get(cmd, ix+1, nothing)
        path = get(cmd, ix+2, nothing)
        xtr = get(cmd, ix+3, nothing)

        if arg === "malloc"
            # Exec benchmark
            realloc_gpu_bumper(40480000)
            malloc_bench(path)
        end

        arg === "filter" &&  filter_bench(path)

        arg === "poller" && poller_bench(path)

        arg === "malloc_count" && malloc_count_test(path)

        arg === "print" && print_bench(path)
        arg === "print2" && print_bench_2(path)

        arg === "des" && des_bench_first(path, xtr)

        arg === "des2" && des_bench_2(path, xtr)

        arg === "despoller" && des_bench_poller(path)
        arg === "despoller2" && des_bench_poller_2(path)

        arg === "des3" && des_bench_total(path)

    elseif type === "exec"

        arg = get(cmd, ix+1, nothing)
        arg2 = get(cmd, ix+2, nothing)
        arg3 = get(cmd, ix+3, nothing)
        arg4 = get(cmd, ix+4, nothing)
        arg5 = get(cmd, ix+5, nothing)

        if arg == "cuda_malloc"
            threads = parse(Int, arg2)
            blocks = parse(Int, arg3)

            manager = arg4 === nothing ?
                CUDA.SimpleAreaManager(div(threads * blocks, 2, RoundUp), 100) :
                eval(Meta.parse(arg4))

            policy = arg5 === nothing ?
                CUDA.NoPolicy :
                eval(Meta.parse(arg5))

            # exec cuda_malloc
            single_cuda_malloc(threads, blocks, manager, policy)
        end

        if arg === "max_count"
            chunks = parse(Int, arg2)
            print_maximum_cuda_mallocs(chunks)
        end

    end
end

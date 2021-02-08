# test runner

using Pkg
Pkg.add(url="https://github.com/maleadt/XUnit.jl", rev="tb/for_loop")

# parse some command-line arguments
function extract_flag!(args, flag, default=nothing)
    for f in args
        if startswith(f, flag)
            # Check if it's just `--flag` or if it's `--flag=foo`
            if f != flag
                val = split(f, '=')[2]
                if default !== nothing && !(typeof(default) <: AbstractString)
                  val = parse(typeof(default), val)
                end
            else
                val = default
            end

            # Drop this value from our args
            filter!(x -> x != f, args)
            return (true, val)
        end
    end
    return (false, default)
end
do_help, _ = extract_flag!(ARGS, "--help")
if do_help
    println("""
        Usage: runtests.jl [--help] [--jobs=N] [TESTS...]

               --help             Show this text.
               --thorough         Don't allow skipping tests that are not supported.
               --mode=single      How to perform tests concurrently (default: distributed).
                     =parallel      The number of concurrently-executed jobs
                     =distributed   is determined by `Threads.nthreads()`.
               --memcheck[=tool]  Run the tests under `cuda-memcheck`.

               Remaining arguments are regular expressions that filter the tests to execute.
               For example, pass 'CUDA/compiler' to only run the compiler tests,
               or '-CUDA/GPUArrays' to skip tests from GPUArrays.jl""")
    exit(0)
end
ORIGINAL_ARGS = copy(ARGS)
_, jobs = extract_flag!(ARGS, "--jobs", Threads.nthreads())
_, mode = extract_flag!(ARGS, "--mode", "distributed")
do_memcheck, memcheck_tool = extract_flag!(ARGS, "--memcheck", "memcheck")
do_thorough, _ = extract_flag!(ARGS, "--thorough")

using XUnit
runtests("tests_early.jl", ARGS...)

using CUDA
@info "System information:\n" * sprint(io->CUDA.versioninfo(io))

# determine the flags used to launch this test process
exeflags = Base.julia_cmd()
filter!(exeflags.exec) do c
    return !(startswith(c, "--depwarn") || startswith(c, "--check-bounds"))
end
push!(exeflags.exec, "--check-bounds=yes")
push!(exeflags.exec, "--startup-file=no")
push!(exeflags.exec, "--depwarn=yes")
if Base.JLOptions().project != C_NULL
    push!(exeflags.exec, "--project=$(unsafe_string(Base.JLOptions().project))")
end
exename = popfirst!(exeflags.exec)

if do_memcheck
    memcheck = CUDA.memcheck()
    @info "Running under $(readchomp(`$memcheck --version`))"
    exename = `$memcheck --tool $memcheck_tool $exename`

    if mode != "distributed"
        # re-launch
        testflags = filter(ORIGINAL_ARGS) do arg
            !startswith(arg, "--memcheck")
        end
        run(`$exename $exeflags $(@__FILE__) $testflags`)
        exit()
    end
end

using Distributed
if mode == "distributed"
    # set-up worker processes (in a proper test environment)
    function addworker(X; kwargs...)
        withenv("JULIA_NUM_THREADS" => 1, "OPENBLAS_NUM_THREADS" => 1) do
            addprocs(X; exename=exename, exeflags=exeflags, kwargs...)
        end
    end
    procs = addworker(Threads.nthreads())
    @everywhere procs begin
        using XUnit
        mode = $mode
        do_memcheck = $do_memcheck
        do_thorough = $do_thorough
    end
end

# execute the main test suite
runtests("tests_all.jl", ARGS...)

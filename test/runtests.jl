using Distributed
using Dates
import REPL
using Printf: @sprintf

# parse some command-line arguments
function extract_flag!(args, flag, default=nothing)
    for f in args
        if startswith(f, flag)
            # Check if it's just `--flag` or if it's `--flag=foo`
            if f != flag
                val = split(f, '=')[2]
                if default !== nothing
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
        Usage: runtests.jl [--help] [--list] [--jobs=N] [TESTS...]

               --help           Show this text.
               --list           List all available tests.
               --jobs=N         Launch `N` process to perform tests.
                                Defaults to `Threads.nthreads()`.
               --memcheck       Run the tests under `cuda-memcheck`.
               --snoop=FILE     Snoop on compiled methods and save to `FILE`.""")
    exit(0)
end
_, jobs = extract_flag!(ARGS, "--jobs", Threads.nthreads())
do_memcheck, _ = extract_flag!(ARGS, "--memcheck")
do_snoop, snoop_path = extract_flag!(ARGS, "--snoop")

include("setup.jl")     # make sure everything is precompiled

# choose tests
const tests = ["initialization",    # needs to run first
               "cutensor"]          # prioritize slow tests
const test_runners = Dict()
## files in the test folder
for (rootpath, dirs, files) in walkdir(@__DIR__)
  # find Julia files
  filter!(files) do file
    endswith(file, ".jl") && file !== "setup.jl" && file !== "runtests.jl"
  end
  isempty(files) && continue

  # strip extension
  files = map(files) do file
    file[1:end-3]
  end

  # prepend subdir
  subdir = relpath(rootpath, @__DIR__)
  if subdir != "."
    files = map(files) do file
      joinpath(subdir, file)
    end
  end

  append!(tests, files)
  for file in files
    test_runners[file] = ()->include("$(@__DIR__)/$file.jl")
  end
end
## GPUArrays testsuite
for name in keys(TestSuite.tests)
    push!(tests, "gpuarrays/$name")
    test_runners["gpuarrays/$name"] = ()->TestSuite.tests[name](CuArray)
end
unique!(tests)

# parse some more command-line arguments
## --list to list all available tests
do_list, _ = extract_flag!(ARGS, "--list")
if do_list
    println("Available tests:")
    for test in sort(tests)
        println(" - $test")
    end
    exit(0)
end
## the remaining args filter tests
if !isempty(ARGS)
  filter!(tests) do test
    any(arg->startswith(test, arg), ARGS)
  end
end

# check that CI is using the requested toolkit
toolkit_release = CUDA.toolkit_release() # ensure artifacts are downloaded
if parse(Bool, get(ENV, "CI", "false")) && haskey(ENV, "JULIA_CUDA_VERSION")
  @test toolkit_release == VersionNumber(ENV["JULIA_CUDA_VERSION"])
end

# pick a suiteable device
candidates = [(device!(dev);
               (dev=dev,
               cap=capability(dev),
               mem=CUDA.available_memory()))
               for dev in devices()]
## pick a device that is fully supported by our CUDA installation, or tools can fail
## NOTE: we don't reuse target_support which is also bounded by LLVM support,
#        and is used to pick a codegen target regardless of the actual device.
cuda_support = CUDA.cuda_compat()
filter!(x->x.cap in cuda_support.cap, candidates)
isempty(candidates) && error("Could not find any suitable device for this configuration")
## order by available memory, but also by capability if testing needs to be thorough
thorough = parse(Bool, get(ENV, "CI_THOROUGH", "false"))
if thorough
    sort!(candidates, by=x->(x.cap, x.mem))
else
    sort!(candidates, by=x->x.mem)
end
pick = last(candidates)
pick.cap >= v"2.0" || error("The CUDA.jl test suite requires a CUDA device with compute capability 2.0 or higher")
@info("Testing using device $(name(pick.dev)) (compute capability $(pick.cap), $(Base.format_bytes(pick.mem)) available memory) on CUDA driver $(CUDA.version()) and toolkit $(CUDA.toolkit_version())")

# determine tests to skip
const skip_tests = []
has_cudnn() || push!(skip_tests, "cudnn")
if !has_cutensor() || CUDA.version() < v"10.1" || pick.cap < v"7.0"
    push!(skip_tests, "cutensor")
end
is_debug = ccall(:jl_is_debugbuild, Cint, ()) != 0
if VERSION < v"1.4.1" || pick.cap < v"7.0" || (is_debug && VERSION < v"1.5.0-DEV.437")
    push!(skip_tests, "device/wmma")
end
if do_memcheck
    # CUFFT causes internal failures in cuda-memcheck
    push!(skip_tests, "cufft")
    # there's also a bunch of `memcheck || ...` expressions in the tests themselves
end
if Sys.ARCH == :aarch64
    # CUFFT segfaults on ARM
    push!(skip_tests, "cufft")
end
if haskey(ENV, "CI_THOROUGH")
    # we're not allowed to skip tests, so make sure we will mark them as such
    all_tests = copy(tests)
    filter!(!in(skip_tests), tests)
else
    if !isempty(skip_tests)
        @info "Skipping the following tests: $(join(skip_tests, ", "))"
        filter!(!in(skip_tests), tests)
    end
    all_tests = copy(tests)
end

# add workers
const test_exeflags = Base.julia_cmd()
filter!(test_exeflags.exec) do c
    return !(startswith(c, "--depwarn") || startswith(c, "--check-bounds"))
end
push!(test_exeflags.exec, "--check-bounds=yes")
push!(test_exeflags.exec, "--startup-file=no")
push!(test_exeflags.exec, "--depwarn=error")
if Base.JLOptions().project != C_NULL
    push!(test_exeflags.exec, "--project=$(unsafe_string(Base.JLOptions().project))")
end
const test_exename = popfirst!(test_exeflags.exec)
function addworker(X; kwargs...)
    exename = if do_memcheck
        `cuda-memcheck $test_exename`
    else
        test_exename
    end

    procs = addprocs(X; exename=exename, exeflags=test_exeflags,
                        dir=@__DIR__, kwargs...)
    @everywhere procs include("setup.jl")
    procs
end
addworker(min(jobs, length(tests)))

# prepare to snoop on the compiler
if do_snoop
    @info "Writing trace of compiled methods to '$snoop_path'"
    snoop_io = open(snoop_path, "w")
end

# pretty print information about gc and mem usage
testgroupheader = "Test"
workerheader = "(Worker)"
name_align        = maximum([textwidth(testgroupheader) + textwidth(" ") +
                             textwidth(workerheader); map(x -> textwidth(x) +
                             3 + ndigits(nworkers()), tests)])
elapsed_align     = textwidth("Time (s)")
gpu_gc_align      = textwidth("GPU GC (s)")
gpu_percent_align = textwidth("GPU GC %")
gpu_alloc_align   = textwidth("GPU Alloc (MB)")
cpu_gc_align      = textwidth("CPU GC (s)")
cpu_percent_align = textwidth("CPU GC %")
cpu_alloc_align   = textwidth("CPU Alloc (MB)")
rss_align         = textwidth("RSS (MB)")
printstyled(testgroupheader, color=:white)
printstyled(lpad(workerheader, name_align - textwidth(testgroupheader) + 1), " | ", color=:white)
printstyled("Time (s) | GPU GC (s) | GPU GC % | GPU Alloc (MB) | CPU GC (s) | CPU GC % | CPU Alloc (MB) | RSS (MB)\n", color=:white)
print_lock = stdout isa Base.LibuvStream ? stdout.lock : ReentrantLock()
if stderr isa Base.LibuvStream
    stderr.lock = print_lock
end
function print_testworker_stats(test, wrkr, resp)
    @nospecialize resp
    lock(print_lock)
    try
        printstyled(test, color=:white)
        printstyled(lpad("($wrkr)", name_align - textwidth(test) + 1, " "), " | ", color=:white)
        time_str = @sprintf("%7.2f",resp[2])
        printstyled(lpad(time_str, elapsed_align, " "), " | ", color=:white)

        gpu_gc_str = @sprintf("%5.2f", resp[7])
        printstyled(lpad(gpu_gc_str, gpu_gc_align, " "), " | ", color=:white)
        # since there may be quite a few digits in the percentage,
        # the left-padding here is less to make sure everything fits
        gpu_percent_str = @sprintf("%4.1f", 100 * resp[7] / resp[2])
        printstyled(lpad(gpu_percent_str, gpu_percent_align, " "), " | ", color=:white)
        gpu_alloc_str = @sprintf("%5.2f", resp[6] / 2^20)
        printstyled(lpad(gpu_alloc_str, gpu_alloc_align, " "), " | ", color=:white)

        cpu_gc_str = @sprintf("%5.2f", resp[4])
        printstyled(lpad(cpu_gc_str, cpu_gc_align, " "), " | ", color=:white)
        cpu_percent_str = @sprintf("%4.1f", 100 * resp[4] / resp[2])
        printstyled(lpad(cpu_percent_str, cpu_percent_align, " "), " | ", color=:white)
        cpu_alloc_str = @sprintf("%5.2f", resp[3] / 2^20)
        printstyled(lpad(cpu_alloc_str, cpu_alloc_align, " "), " | ", color=:white)

        rss_str = @sprintf("%5.2f", resp[9] / 2^20)
        printstyled(lpad(rss_str, rss_align, " "), "\n", color=:white)
    finally
        unlock(print_lock)
    end
end
global print_testworker_started = (name, wrkr)->begin
    if do_memcheck
        lock(print_lock)
        try
            printstyled(name, color=:white)
            printstyled(lpad("($wrkr)", name_align - textwidth(name) + 1, " "), " |",
                " "^elapsed_align, "started at $(now())\n", color=:white)
        finally
            unlock(print_lock)
        end
    end
end
function print_testworker_errored(name, wrkr)
    lock(print_lock)
    try
        printstyled(name, color=:red)
        printstyled(lpad("($wrkr)", name_align - textwidth(name) + 1, " "), " |",
            " "^elapsed_align, " failed at $(now())\n", color=:red)
    finally
        unlock(print_lock)
    end
end

# run tasks
results = []
all_tasks = Task[]
try
    # Monitor stdin and kill this task on ^C
    # but don't do this on Windows, because it may deadlock in the kernel
    running_tests = Dict{String, DateTime}()
    if !Sys.iswindows() && isa(stdin, Base.TTY)
        t = current_task()
        stdin_monitor = @async begin
            term = REPL.Terminals.TTYTerminal("xterm", stdin, stdout, stderr)
            try
                REPL.Terminals.raw!(term, true)
                while true
                    c = read(term, Char)
                    if c == '\x3'
                        Base.throwto(t, InterruptException())
                        break
                    elseif c == '?'
                        println("Currently running: ")
                        tests = sort(collect(running_tests), by=x->x[2])
                        foreach(tests) do (test, date)
                            println(test, " (running for ", round(now()-date, Minute), ")")
                        end
                    end
                end
            catch e
                isa(e, InterruptException) || rethrow()
            finally
                REPL.Terminals.raw!(term, false)
            end
        end
    end
    @sync begin
        for p in workers()
            @async begin
                push!(all_tasks, current_task())
                while length(tests) > 0
                    test = popfirst!(tests)
                    local resp
                    wrkr = p
                    dev = test=="initialization" ? nothing : pick.dev
                    snoop = do_snoop ? mktemp() : (nothing, nothing)

                    # run the test
                    running_tests[test] = now()
                    try
                        resp = remotecall_fetch(runtests, wrkr, test_runners[test], test, dev, snoop[1])
                    catch e
                        isa(e, InterruptException) && return
                        resp = Any[e]
                    end
                    delete!(running_tests, test)
                    push!(results, (test, resp))

                    # act on the results
                    if resp[1] isa Exception
                        print_testworker_errored(test, wrkr)

                        # the worker encountered some failure, recycle it
                        # so future tests get a fresh environment
                        rmprocs(wrkr, waitfor=30)
                        p = addworker(1)[1]
                    else
                        print_testworker_stats(test, wrkr, resp)
                    end

                    # aggregate the snooped compiler invocations
                    if do_snoop
                        for line in eachline(snoop[2])
                            println(snoop_io, line)
                        end
                        close(snoop[2])
                        rm(snoop[1])
                    end
                end
                if p != 1
                    # Free up memory =)
                    rmprocs(p, waitfor=30)
                end
            end
        end
    end
catch e
    isa(e, InterruptException) || rethrow()
    # If the test suite was merely interrupted, still print the
    # summary, which can be useful to diagnose what's going on
    foreach(task -> begin
            istaskstarted(task) || return
            istaskdone(task) && return
            try
                schedule(task, InterruptException(); error=true)
            catch ex
                @error "InterruptException" exception=ex,catch_backtrace()
            end
        end, all_tasks)
    foreach(wait, all_tasks)
finally
    if @isdefined stdin_monitor
        schedule(stdin_monitor, InterruptException(); error=true)
    end
end

# construct a testset to render the test results
o_ts = Test.DefaultTestSet("Overall")
Test.push_testset(o_ts)
completed_tests = Set{String}()
for (testname, (resp,)) in results
    push!(completed_tests, testname)
    if isa(resp, Test.DefaultTestSet)
        Test.push_testset(resp)
        Test.record(o_ts, resp)
        Test.pop_testset()
    elseif isa(resp, Tuple{Int,Int})
        fake = Test.DefaultTestSet(testname)
        for i in 1:resp[1]
            Test.record(fake, Test.Pass(:test, nothing, nothing, nothing))
        end
        for i in 1:resp[2]
            Test.record(fake, Test.Broken(:test, nothing))
        end
        Test.push_testset(fake)
        Test.record(o_ts, fake)
        Test.pop_testset()
    elseif isa(resp, RemoteException) && isa(resp.captured.ex, Test.TestSetException)
        println("Worker $(resp.pid) failed running test $(testname):")
        Base.showerror(stdout, resp.captured)
        println()
        fake = Test.DefaultTestSet(testname)
        for i in 1:resp.captured.ex.pass
            Test.record(fake, Test.Pass(:test, nothing, nothing, nothing))
        end
        for i in 1:resp.captured.ex.broken
            Test.record(fake, Test.Broken(:test, nothing))
        end
        for t in resp.captured.ex.errors_and_fails
            Test.record(fake, t)
        end
        Test.push_testset(fake)
        Test.record(o_ts, fake)
        Test.pop_testset()
    else
        if !isa(resp, Exception)
            resp = ErrorException(string("Unknown result type : ", typeof(resp)))
        end
        # If this test raised an exception that is not a remote testset exception,
        # i.e. not a RemoteException capturing a TestSetException that means
        # the test runner itself had some problem, so we may have hit a segfault,
        # deserialization errors or something similar.  Record this testset as Errored.
        fake = Test.DefaultTestSet(testname)
        Test.record(fake, Test.Error(:test_error, testname, nothing, Any[(resp, [])], LineNumberNode(1)))
        Test.push_testset(fake)
        Test.record(o_ts, fake)
        Test.pop_testset()
    end
end
for test in all_tests
    (test in completed_tests) && continue
    fake = Test.DefaultTestSet(test)
    Test.record(fake, Test.Error(:test_interrupted, test, nothing,
                                    [("skipped", [])], LineNumberNode(1)))
    Test.push_testset(fake)
    Test.record(o_ts, fake)
    Test.pop_testset()
end
println()
Test.print_test_results(o_ts, 1)
if !o_ts.anynonpass
    println("    \033[32;1mSUCCESS\033[0m")
else
    println("    \033[31;1mFAILURE\033[0m\n")
    Test.print_test_errors(o_ts)
    throw(Test.FallbackTestSetException("Test run finished with errors"))
end


using Distributed
using Dates
import REPL
using Printf: @sprintf
using TimerOutputs

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
        Usage: runtests.jl [--help] [--list] [--jobs=N] [TESTS...]

               --help             Show this text.
               --list             List all available tests.
               --thorough         Don't allow skipping tests that are not supported.
               --quickfail        Fail the entire run as soon as a single test errored.
               --jobs=N           Launch `N` processes to perform tests (default: Threads.nthreads()).
               --gpus=N           Expose `N` GPUs to test processes (default: 1).
               --sanitize[=tool]  Run the tests under `compute-sanitizer`.
               --snoop=FILE       Snoop on compiled methods and save to `FILE`.

               Remaining arguments filter the tests that will be executed.""")
    exit(0)
end
set_jobs, jobs = extract_flag!(ARGS, "--jobs", Threads.nthreads())
do_sanitize, sanitize_tool = extract_flag!(ARGS, "--sanitize", "memcheck")
do_snoop, snoop_path = extract_flag!(ARGS, "--snoop")
do_thorough, _ = extract_flag!(ARGS, "--thorough")
do_quickfail, _ = extract_flag!(ARGS, "--quickfail")

if !set_jobs && jobs == 1
    @warn """You are running the CUDA.jl test suite with only a single thread; this will take a long time.
           Consider launching Julia with `--threads auto` to run tests in parallel."""
end

include("setup.jl")     # make sure everything is precompiled
_, gpus = extract_flag!(ARGS, "--gpus", ndevices())

# choose tests
const tests = ["initialization"]    # needs to run first
const test_runners = Dict()
## GPUArrays testsuite
for name in keys(TestSuite.tests)
    push!(tests, "gpuarrays$(Base.Filesystem.path_separator)$name")
    test_runners["gpuarrays$(Base.Filesystem.path_separator)$name"] = ()->TestSuite.tests[name](CuArray)
end
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
  sort(files; by=(file)->stat("$(@__DIR__)/$file.jl").size, rev=true) # large (slow) tests first
  for file in files
    test_runners[file] = ()->include("$(@__DIR__)/$file.jl")
  end
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
## no options should remain
optlike_args = filter(startswith("-"), ARGS)
if !isempty(optlike_args)
    error("Unknown test options `$(join(optlike_args, " "))` (try `--help` for usage instructions)")
end
## the remaining args filter tests
if !isempty(ARGS)
  filter!(tests) do test
    any(arg->startswith(test, arg), ARGS)
  end
end

# check that CI is using the requested toolkit
toolkit_release = CUDA.toolkit_release() # ensure artifacts are downloaded
if CUDA.getenv("CI", false) && haskey(ENV, "JULIA_CUDA_VERSION")
  @test toolkit_release == VersionNumber(ENV["JULIA_CUDA_VERSION"])
end

# find suitable devices
@info "System information:\n" * sprint(io->CUDA.versioninfo(io))
candidates = []
for (index,dev) in enumerate(devices())
    # fetch info that doesn't require a context
    id = deviceid(dev)
    mig = CUDA.uuid(dev) != CUDA.parent_uuid(dev)
    uuid = CUDA.uuid(dev)
    name = CUDA.name(dev)
    cap = capability(dev)

    mem = try
        device!(dev)
        mem = CUDA.available_memory()
        # immediately reset the device. this helps to reduce memory usage,
        # and is needed for systems that only provide exclusive access to the GPUs
        CUDA.device_reset!()
        mem
    catch err
        if isa(err, OutOfGPUMemoryError)
            # the device doesn't even have enough memory left to instantiate a context...
            0
        else
            rethrow()
        end
    end

    push!(candidates, (; id, uuid, mig, name, cap, mem))

    # NOTE: we don't use NVML here because it doesn't respect CUDA_VISIBLE_DEVICES
end
## only consider devices that are fully supported by our CUDA toolkit, or tools can fail.
## NOTE: we don't reuse supported_toolchain() which is also bounded by LLVM support,
#        and is used to pick a codegen target regardless of the actual device.
cuda_support = CUDA.cuda_compat()
filter!(x->x.cap in cuda_support.cap, candidates)
## only consider recent devices if we want testing to be thorough
if do_thorough
    filter!(x->x.cap >= v"6.0", candidates)
end
isempty(candidates) && error("Could not find any suitable device for this configuration")
## order by available memory, but also by capability if testing needs to be thorough
sort!(candidates, by=x->x.mem)
## apply
picks = reverse(candidates[end-gpus+1:end])   # best GPU first
ENV["CUDA_VISIBLE_DEVICES"] = join(map(pick->"$(pick.mig ? "MIG" : "GPU")-$(pick.uuid)", picks), ",")
@info "Testing using $(length(picks)) device(s): " * join(map(pick->"$(pick.id). $(pick.name) (UUID $(pick.uuid))", picks), ", ")

# determine tests to skip
skip_tests = []
has_cudnn() || push!(skip_tests, "cudnn")
has_cusolvermg() || push!(skip_tests, "cusolver/multigpu")
has_nvml() || push!(skip_tests, "nvml")
if !has_cutensor() || first(picks).cap < v"6.0"
    push!(skip_tests, "cutensor")
end
has_cutensornet() || push!(skip_tests, "cutensornet")
has_custatevec() || push!(skip_tests, "custatevec")
if do_sanitize
    # XXX: some library tests fail under compute-sanitizer
    append!(skip_tests, ["cutensor"])
    # XXX: others take absurdly long
    push!(skip_tests, "cusolver")
    # XXX: these hang for some reason
    push!(skip_tests, "sorting")
end
if first(picks).cap < v"7.0"
    push!(skip_tests, "device/intrinsics/wmma")
end
if Sys.ARCH == :aarch64
    # CUFFT segfaults on ARM
    push!(skip_tests, "cufft")
end
if VERSION < v"1.6.1-"
    push!(skip_tests, "device/random")
end
for (i, test) in enumerate(skip_tests)
    # we find tests by scanning the file system, so make sure the path separator matches
    skip_tests[i] = replace(test, '/'=>Base.Filesystem.path_separator)
end
# skip_tests is a list of patterns, expand it to actual tests we were going to run
skip_tests = filter(test->any(skip->occursin(skip,test), skip_tests), tests)
if do_thorough
    # we're not allowed to skip tests, so make sure we will mark them as such
    all_tests = copy(tests)
    if !isempty(skip_tests)
        @error "Skipping the following tests: $(join(skip_tests, ", "))"
        filter!(!in(skip_tests), tests)
    end
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
push!(test_exeflags.exec, "--depwarn=yes")
if Base.JLOptions().project != C_NULL
    push!(test_exeflags.exec, "--project=$(unsafe_string(Base.JLOptions().project))")
end
const test_exename = popfirst!(test_exeflags.exec)
function addworker(X; kwargs...)
    exename = if do_sanitize
        `$(CUDA.compute_sanitizer_cmd(sanitize_tool)) $test_exename`
    else
        test_exename
    end

    withenv("JULIA_NUM_THREADS" => 1, "OPENBLAS_NUM_THREADS" => 1) do
        procs = addprocs(X; exename=exename, exeflags=test_exeflags, kwargs...)
        @everywhere procs include($(joinpath(@__DIR__, "setup.jl")))
        procs
    end
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
gc_align      = textwidth("GC (s)")
percent_align = textwidth("GC %")
alloc_align   = textwidth("Alloc (MB)")
rss_align     = textwidth("RSS (MB)")
printstyled(" "^(name_align + textwidth(testgroupheader) - 3), " | ")
printstyled("         | ---------------- GPU ---------------- | ---------------- CPU ---------------- |\n", color=:white)
printstyled(testgroupheader, color=:white)
printstyled(lpad(workerheader, name_align - textwidth(testgroupheader) + 1), " | ", color=:white)
printstyled("Time (s) | GC (s) | GC % | Alloc (MB) | RSS (MB) | GC (s) | GC % | Alloc (MB) | RSS (MB) |\n", color=:white)
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
        printstyled(lpad(gpu_gc_str, gc_align, " "), " | ", color=:white)
        # since there may be quite a few digits in the percentage,
        # the left-padding here is less to make sure everything fits
        gpu_percent_str = @sprintf("%4.1f", 100 * resp[7] / resp[2])
        printstyled(lpad(gpu_percent_str, percent_align, " "), " | ", color=:white)
        gpu_alloc_str = @sprintf("%5.2f", resp[6] / 2^20)
        printstyled(lpad(gpu_alloc_str, alloc_align, " "), " | ", color=:white)

        gpu_rss_str = ismissing(resp[10]) ? "N/A" : @sprintf("%5.2f", resp[10] / 2^20)
        printstyled(lpad(gpu_rss_str, rss_align, " "), " | ", color=:white)

        cpu_gc_str = @sprintf("%5.2f", resp[4])
        printstyled(lpad(cpu_gc_str, gc_align, " "), " | ", color=:white)
        cpu_percent_str = @sprintf("%4.1f", 100 * resp[4] / resp[2])
        printstyled(lpad(cpu_percent_str, percent_align, " "), " | ", color=:white)
        cpu_alloc_str = @sprintf("%5.2f", resp[3] / 2^20)
        printstyled(lpad(cpu_alloc_str, alloc_align, " "), " | ", color=:white)

        cpu_rss_str = @sprintf("%5.2f", resp[9] / 2^20)
        printstyled(lpad(cpu_rss_str, rss_align, " "), " |\n", color=:white)
    finally
        unlock(print_lock)
    end
end
global print_testworker_started = (name, wrkr)->begin
    if do_sanitize
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
t0 = now()
results = []
all_tasks = Task[]
timings = TimerOutput[]
try
    # Monitor stdin and kill this task on ^C
    # but don't do this on Windows, because it may deadlock in the kernel
    t = current_task()
    running_tests = Dict{String, DateTime}()
    if !Sys.iswindows() && isa(stdin, Base.TTY)
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
        function recycle_worker(p)
            if isdefined(CUDA, :to)
                to = remotecall_fetch(p) do
                    CUDA.to
                end
                push!(timings, to)
            end

            rmprocs(p, waitfor=30)

            return nothing
        end

        for p in workers()
            @async begin
                push!(all_tasks, current_task())
                while length(tests) > 0
                    test = popfirst!(tests)

                    # sometimes a worker failed, and we need to spawn a new one
                    if p === nothing
                        p = addworker(1)[1]
                    end
                    wrkr = p

                    local resp
                    snoop = do_snoop ? mktemp() : (nothing, nothing)

                    # tests that muck with the context should not be timed with CUDA events,
                    # since they won't be valid at the end of the test anymore.
                    time_source = in(test, ["initialization", "examples", "exceptions"]) ? :julia : :cuda

                    # run the test
                    running_tests[test] = now()
                    try
                        resp = remotecall_fetch(runtests, wrkr, test_runners[test], test, time_source, snoop[1])
                    catch e
                        isa(e, InterruptException) && return
                        resp = Any[e]
                    end
                    delete!(running_tests, test)
                    push!(results, (test, resp))

                    # act on the results
                    if resp[1] isa Exception
                        print_testworker_errored(test, wrkr)
                        do_quickfail && Base.throwto(t, InterruptException())

                        # the worker encountered some failure, recycle it
                        # so future tests get a fresh environment
                        p = recycle_worker(p)
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

                if p !== nothing
                    recycle_worker(p)
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
    for t in all_tasks
        # NOTE: we can't just wait, but need to discard the exception,
        #       because the throwto for --quickfail also kills the worker.
        try
            wait(t)
        catch e
            showerror(stderr, e)
        end
    end
finally
    if @isdefined stdin_monitor
        schedule(stdin_monitor, InterruptException(); error=true)
    end
end
t1 = now()
elapsed = canonicalize(Dates.CompoundPeriod(t1-t0))
println("Testing finished in $elapsed")

# report work timings
if isdefined(CUDA, :to)
    println()
    for to in timings
        TimerOutputs.merge!(CUDA.to, to)
    end
    TimerOutputs.complement!(CUDA.to)
    show(CUDA.to, sortby=:name)
    println()
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
            if VERSION >= v"1.7-"
                Test.record(fake, Test.Pass(:test, nothing, nothing, nothing, nothing))
            else
                Test.record(fake, Test.Pass(:test, nothing, nothing, nothing))
            end
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
            if VERSION >= v"1.7-"
                Test.record(fake, Test.Pass(:test, nothing, nothing, nothing, nothing))
            else
                Test.record(fake, Test.Pass(:test, nothing, nothing, nothing))
            end
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
        Test.record(fake, Test.Error(:nontest_error, testname, nothing, Any[(resp, [])], LineNumberNode(1)))
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

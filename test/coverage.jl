#!/usr/bin/env julia

# This script will gather coverage information with the currently active julia,
# and will process it with Coverage.jl using the system-wide Julia at /usr/bin
# (this for compatibility reasons, but might break if .cov parsing changes).

cmd = Base.julia_cmd()

system_cmd = "/usr/bin/julia"
system_env = copy(ENV)
for (k,v) in system_env
    if startswith(k, "JULIA_")
        delete!(system_env, k)
    end
end

# make sure Coverage.jl is available on the clean Julia
run(setenv(`$system_cmd -E "using Coverage"`, system_env))


#
# Measure
#

dir = dirname(@__FILE__)
root = "$dir/../"
cd(root)

run(`$cmd --compilecache=no --inline=no --code-coverage=user "test/runtests.jl"`)

ENV["DEBUG"] = 1
run(`$cmd --compilecache=no --inline=no --code-coverage=user "test/runtests.jl"`)

ENV["TRACE"] = 1
run(`$cmd --compilecache=no --inline=no --code-coverage=user "test/runtests.jl"`)


#
# Submit (with the system Julia)
#

let code = """
    using Coverage

    coverage = process_folder()

    codecov = haskey(ENV, "USE_CODECOV")
    lcov    = haskey(ENV, "USE_LCOV")
    if !codecov && !lcov
        codecov = true
    end

    if codecov
        ENV["REPO_TOKEN"] = "d8b27424-577e-4851-a490-9e5459b5d787"
        Codecov.submit_token(coverage)
    end

    if lcov
        isdir("test/coverage") || mkdir("test/coverage")
        LCOV.writefile("test/coverage/lcov.info", coverage)
        run(`genhtml test/coverage/lcov.info -o test/coverage/html`)
        println("Done! Open your browser at file://$dir/coverage/html/index.html")
    end

    clean_folder(".")
    """

    run(setenv(`$system_cmd -E "$code"`, system_env))
end

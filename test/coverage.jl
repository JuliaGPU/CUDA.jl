#!/usr/bin/env julia

codecov = haskey(ENV, "USE_CODECOV")
lcov    = haskey(ENV, "USE_LCOV")
if !codecov && !lcov
    codecov = true
end

using Coverage

dir = dirname(@__FILE__)
root = "$dir/../"
cd(root)

# TODO: --inline=no

run(`julia --compilecache=no --code-coverage=user "test/runtests.jl"`)

ENV["DEBUG"] = 1
run(`julia --compilecache=no --code-coverage=user "test/runtests.jl"`)

ENV["TRACE"] = 1
run(`julia --compilecache=no --code-coverage=user "test/runtests.jl"`)

coverage = process_folder()

if codecov
    ENV["REPO_TOKEN"] = "12328f4d-3b60-46df-a3c8-4314e682c414"
    Codecov.submit_token(coverage)
end

if lcov
    isdir("test/coverage") || mkdir("test/coverage")
    LCOV.writefile("test/coverage/lcov.info", coverage)
    run(`genhtml test/coverage/lcov.info -o test/coverage/html`)
    println("Done! Open your browser at file://$dir/coverage/html/index.html")
end

clean_folder(".")

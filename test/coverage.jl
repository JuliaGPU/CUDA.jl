#!/usr/bin/env julia

using Coverage

ENV["REPO_TOKEN"] = "12328f4d-3b60-46df-a3c8-4314e682c414"

dir, _ = splitdir(Base.source_path())
root = "$dir/../"
cd(root)

ENV["TRACE"] = 1
# TODO: --inline=no
run(`julia --compilecache=no --code-coverage=user "test/runtests.jl"`)

coverage = process_folder()

Codecov.submit_token(coverage)

clean_folder(".")

#!/usr/bin/env julia

using Coverage

ENV["REPO_TOKEN"] = "d8b27424-577e-4851-a490-9e5459b5d787"

dir, _ = splitdir(Base.source_path())
root = "$dir/../"
cd(root)

ENV["TRACE"] = 1
run(`julia --compilecache=no --inline=no --code-coverage=user "test/runtests.jl"`)

coverage = process_folder()

Codecov.submit_token(coverage)

clean_folder(".")

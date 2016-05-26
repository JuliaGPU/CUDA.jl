#!/usr/bin/env julia

dir, _ = splitdir(Base.source_path())
root = "$dir/../"
cd(root)

# TODO: --inline=no
run(`julia --code-coverage=user "test/runtests.jl"`)

using Coverage
coverage = process_folder()
LCOV.writefile("coverage/lcov.info", coverage)
clean_folder(".")

run(`genhtml coverage/lcov.info -o coverage/html`)
println("Done! Open your browser at file://$dir/html/index.html")

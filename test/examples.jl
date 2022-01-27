# NVIDIA bug 3263616: compute-sanitizer crashes when generating host backtraces,
#                     but --show-backtrace=no does not survive execve.
@not_if_sanitize begin

function find_sources(path::String, sources=String[])
    if isdir(path)
        for entry in readdir(path)
            find_sources(joinpath(path, entry), sources)
        end
    elseif endswith(path, ".jl")
        push!(sources, path)
    end
    sources
end

examples_dir = joinpath(@__DIR__, "..", "examples")
examples = find_sources(examples_dir)
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

cd(examples_dir) do
    global examples
    examples = relpath.(examples, Ref(examples_dir))
    @testset for example in examples
        proc, out, err = julia_exec(`$example`)
        isempty(err) || println(err)
        @test success(proc)
    end
end

end

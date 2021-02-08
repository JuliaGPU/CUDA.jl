@testcase "examples" begin

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

for example in examples
    example_name = first(splitext(relpath(example, examples_dir)))
    @testcase "$example_name" begin
        @in_module quote
            cd(dirname($example)) do
                include($example)
            end
        end
    end
end

end

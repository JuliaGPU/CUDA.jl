@testset "examples" begin

function find_sources(dir)
    sources = String[]
    for entry in readdir(dir)
        path = joinpath(dir, entry)
        if isdir(path)
            append!(sources, find_sources(path))
        elseif endswith(path, ".jl")
            push!(sources, path)
        end
    end
    return sources
end

examples_dir = joinpath(@__DIR__, "..", "examples")
examples = find_sources(examples_dir)
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

for example in examples
    id = relpath(example, examples_dir)
    @eval begin
        @testset $id begin
            file = $example
            cmd = julia_cmd(`$file`)
            @test success(pipeline(cmd, stderr=stderr))
        end
    end
end

end

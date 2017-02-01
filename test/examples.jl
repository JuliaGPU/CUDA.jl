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

examples = find_sources(joinpath(@__DIR__, "..", "examples"))
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

for example in examples
    run(`$(Base.julia_cmd()) $example`)
end

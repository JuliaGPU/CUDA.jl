# The packages of the CUDA.jl monorepo, as listed in the root Project.toml's `[workspace]`.
# Shared by the release scripts so that the roster is not maintained by hand.

using TOML

"""
    workspace_packages(read_project) -> Vector{@NamedTuple{name, subdir, project}}

List the root package and every `[workspace]` member that is a package (i.e., not an
environment like `test/` or `docs/`). `read_project(subdir)` returns the parsed
`Project.toml` in `subdir` (`""` for the root), or `nothing` if there is none.
"""
function workspace_packages(read_project)
    root = read_project("")
    packages = [(; name=root["name"], subdir="", project=root)]
    for subdir in root["workspace"]["projects"]
        project = read_project(subdir)
        (project === nothing || !haskey(project, "uuid")) && continue
        push!(packages, (; name=project["name"], subdir, project))
    end
    return packages
end

"""
    registration_waves(packages) -> Vector{Vector{String}}

Group packages into waves that can be registered together: every package only depends on
packages from earlier waves. Weak dependencies are ignored, since CUDACore has an extension
on CUDA itself.
"""
function registration_waves(packages)
    internal = Set(p.name for p in packages)
    deps = Dict(p.name => intersect(keys(get(p.project, "deps", Dict())), internal)
                for p in packages)
    waves = Vector{String}[]
    done = Set{String}()
    while length(done) < length(packages)
        wave = sort([name for (name, d) in deps if name ∉ done && d ⊆ done])
        isempty(wave) && error("dependency cycle among $(setdiff(internal, done))")
        push!(waves, wave)
        union!(done, wave)
    end
    return waves
end

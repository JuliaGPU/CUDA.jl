using CUDA.Deps: @initialize_ref, generic_artifact, artifact_library, find_library,
                 LocalToolkit, ArtifactToolkit, toolkit

using Libdl

export libcutensornet, has_cutensornet

const __libcutensornet = Ref{Union{String,Nothing}}()
function libcutensornet(; throw_error::Bool=true)
    path = @initialize_ref __libcutensornet begin
        # CUTENSORNET depends on CUTENSOR
        CUTENSOR.libcutensor(throw_error=throw_error)

        if CUDA.runtime_version() < v"11"
            # XXX: bound this using tags in the Artifact.toml?
            nothing
        else
            find_cutensornet(toolkit(), "cutensornet", v"0.1")
        end
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUTENSORNET is missing.")
    end
    return path
end
has_cutensornet() = CUTENSOR.has_cutensor() && libcutensornet(throw_error=false) !== nothing

function find_cutensornet(cuda::ArtifactToolkit, name, version)
    artifact_dir = generic_artifact("cuQuantum")
    if artifact_dir === nothing
        return nothing
    end
    path = artifact_library(artifact_dir, name, [version])

    @debug "Using CUTENSORNET library $name from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_cutensornet(cuda::LocalToolkit, name, version)
    path = find_library(name, [version]; locations=cuda.dirs)
    if path === nothing
        return nothing
    end

    @debug "Using local CUTENSORNET library $name at $(path)"
    Libdl.dlopen(path)
    return path
end

using CUDA.Deps: @initialize_ref, generic_artifact, artifact_library, find_library,
                 LocalToolkit, ArtifactToolkit, toolkit

using Libdl

export libcustatevec, has_custatevec

const __libcustatevec = Ref{Union{String,Nothing}}()
function libcustatevec(; throw_error::Bool=true)
    path = @initialize_ref __libcustatevec begin

        if CUDA.runtime_version() < v"11"
            # XXX: bound this using tags in the Artifact.toml?
            nothing
        else
           find_custatevec(toolkit(), "custatevec", v"0.1")
        end
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUSTATEVEC is missing.")
    end
    return path
end
has_custatevec() = libcustatevec(throw_error=false) !== nothing

function find_custatevec(cuda::ArtifactToolkit, name, version)
    artifact_dir = generic_artifact("cuQuantum")
    if artifact_dir === nothing
        return nothing
    end
    path = artifact_library(artifact_dir, name, [version])

    @debug "Using CUSTATEVEC library $name from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_custatevec(cuda::LocalToolkit, name, version)
    path = find_library(name, [version]; locations=cuda.dirs)
    if path === nothing
        return nothing
    end

    @debug "Using local CUSTATEVEC library $name at $(path)"
    Libdl.dlopen(path)
    return path
end

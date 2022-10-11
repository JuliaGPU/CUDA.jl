using CUDA.Deps: @initialize_ref, cuda_artifact, artifact_library, find_library,
                 LocalToolkit, ArtifactToolkit, toolkit,
                 libcublas, libcudart

import Libdl

export libcutensor, libcutensormg, has_cutensor, has_cutensormg

const __libcutensor = Ref{Union{String,Nothing}}()
function libcutensor(; throw_error::Bool=true)
    path = @initialize_ref __libcutensor begin
        # CUTENSOR depends on CUBLAS
        libcublas()

        find_cutensor(toolkit(), "cutensor", v"1")
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUTENSOR is missing.")
    end
    path
end
has_cutensor() = libcutensor(throw_error=false) !== nothing

const __libcutensormg = Ref{Union{String,Nothing}}()
function libcutensormg(; throw_error::Bool=true)
    path = @initialize_ref __libcutensor begin
        # CUTENSORMg additionally depends on CUDARt
        libcudart()

        if CUTENSOR.version() < v"1.4"
            nothing
        else
            find_cutensor(toolkit(), "cutensorMg", v"1")
        end
    end
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUTENSORMg is missing.")
    end
    path
end
has_cutensormg() = libcutensormg(throw_error=false) !== nothing

function find_cutensor(cuda::ArtifactToolkit, name, version)
    artifact_dir = cuda_artifact("CUTENSOR", cuda.release)
    if artifact_dir === nothing
        return nothing
    end
    path = artifact_library(artifact_dir, name, [version])

    @debug "Using CUTENSOR library $name from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_cutensor(cuda::LocalToolkit, name, version)
    path = find_library(name, [version]; locations=cuda.dirs)
    if path === nothing
        return nothing
    end

    @debug "Using local CUTENSOR library $name at $(path)"
    Libdl.dlopen(path)
    return path
end

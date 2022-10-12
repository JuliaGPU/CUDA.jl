using CUDA.Deps: @initialize_ref, libcublas, cuda_artifact, artifact_library, find_library,
                 LocalToolkit, ArtifactToolkit, toolkit

import Libdl

export libcudnn, has_cudnn

const __libcudnn = Ref{Union{String,Nothing}}()
function libcudnn(; throw_error::Bool=true)
    path = @initialize_ref __libcudnn begin
        # CUDNN depends on CUBLAS
        libcublas()

        find_cudnn(toolkit(), v"8")
    end __runtime_init__()
    if path === nothing && throw_error
        error("This functionality is unavailabe as CUDNN is missing.")
    end
    path
end
has_cudnn() = libcudnn(throw_error=false) !== nothing

function find_cudnn(cuda::ArtifactToolkit, version)
    artifact_dir = cuda_artifact("CUDNN", cuda.release)
    if artifact_dir === nothing
        return nothing
    end
    path = artifact_library(artifact_dir, "cudnn", [version])

    # HACK: eagerly open CUDNN sublibraries to avoid dlopen discoverability issues
    for sublibrary in ("ops_infer", "ops_train",
                       "cnn_infer", "cnn_train",
                       "adv_infer", "adv_train")
        sublibrary_path = artifact_library(artifact_dir, "cudnn_$(sublibrary)", [version])
        Libdl.dlopen(sublibrary_path)
    end

    @debug "Using CUDNN from an artifact at $(artifact_dir)"
    Libdl.dlopen(path)
    return path
end

function find_cudnn(cuda::LocalToolkit, version)
    path = find_library("cudnn", [version]; locations=cuda.dirs)
    if path === nothing
        return nothing
    end

    # with a local CUDNN version, we shouldn't need to eagerly open sublibraries,
    # as they are expected to be globally discoverable next to libcudnn.so

    @debug "Using local CUDNN at $(path)"
    Libdl.dlopen(path)
    return path
end

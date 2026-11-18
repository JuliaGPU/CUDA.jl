# run-time environment of the HostJIT compiler

# headers that the embedded Clang needs, shipped with the JLL
function configure_hostjit!()
    share = joinpath(CCCL_C_jll.artifact_dir, "share", "cccl")
    get!(ENV, "HOSTJIT_INCLUDE_PATH", share)
    get!(ENV, "HOSTJIT_CLANG_PATH", joinpath(share, "clang"))

    # libnvcc links nvJitLink by soname; make sure it resolves to the one CUDA.jl uses
    Libdl.dlopen(CUDACore.CUDA_Compiler.libnvJitLink_path)

    return
end

# the CUB, Thrust and libcu++ headers the library was built against
cccl_include_path() = joinpath(CCCL_C_jll.artifact_dir, "include")

"""
    CCCL.toolkit_path() -> String

The CUDA toolkit whose headers, `libdevice` and CUDA runtime library HostJIT compiles and
links the algorithms against. Taken from `CUDA_PATH` or `CUDA_HOME`, or a toolkit in a
standard location.

The JLLs CUDA.jl uses do not ship the CUDA runtime headers, so this currently needs a local
CUDA toolkit.
"""
function toolkit_path()
    candidates = String[]
    for var in ("CUDA_PATH", "CUDA_HOME")
        haskey(ENV, var) && push!(candidates, ENV[var])
    end
    append!(candidates, ["/usr/local/cuda", "/opt/cuda"])
    for dir in candidates
        if isfile(joinpath(dir, "include", "cuda_runtime.h")) &&
           isfile(joinpath(dir, "nvvm", "libdevice", "libdevice.10.bc"))
            return dir
        end
    end
    error("CCCL needs a local CUDA toolkit (looked in $(join(candidates, ", "))); " *
          "set CUDA_PATH to point to one")
end

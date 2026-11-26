# run-time environment of the HostJIT compiler

# headers that the embedded Clang needs, shipped with the JLL
function configure_hostjit!()
    share = joinpath(CCCL_C_jll.artifact_dir, "share", "cccl")
    get!(ENV, "HOSTJIT_INCLUDE_PATH", share)
    get!(ENV, "HOSTJIT_CLANG_PATH", joinpath(share, "clang"))

    # libnvcc links nvJitLink by soname; make sure it resolves to the one CUDA.jl uses
    path = nvjitlink_path()
    handle = path === nothing ? nothing : Libdl.dlopen(path; throw_error=false)
    if handle === nothing
        @warn """CCCL could not load nvJitLink from $(something(path, toolkit_path())).
                 CCCL.functional() will return false."""
        return false
    end

    # the PTX HostJIT generates declares CUB's shared temporary storage `.weak`, which the
    # nvJitLink of CUDA 13.0 to 13.4 rejects as "bad input" (CUDA 12.9 and 13.5 accept it).
    # The result is a cubin, so the driver only needs to support the same major version.
    major, minor = Ref{Cint}(), Ref{Cint}()
    ccall(Libdl.dlsym(handle, :nvJitLinkVersion), Cint, (Ptr{Cint}, Ptr{Cint}), major, minor)
    nvjitlink = VersionNumber(major[], minor[])
    if v"13" <= nvjitlink < v"13.5"
        @warn """CCCL needs nvJitLink from CUDA 13.5 or later, but $path is from CUDA $(major[]).$(minor[]).
                 CCCL.functional() will return false."""
        return false
    end

    return true
end

# the CUDA_Compiler JLL ships nvJitLink, but CUDA_Runtime_Discovery, used with a local
# toolkit, does not discover it, so take it from the toolkit HostJIT compiles against
function nvjitlink_path()
    compiler = CUDACore.CUDA_Compiler
    isdefined(compiler, :libnvJitLink_path) && return compiler.libnvJitLink_path

    dir = toolkit_path()
    path = Libdl.find_library("libnvJitLink", [joinpath(dir, "lib64"), joinpath(dir, "lib")])
    return isempty(path) ? nothing : path
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
    dir = find_toolkit()
    if dir === nothing
        error("CCCL needs a local CUDA toolkit (looked in $(join(toolkit_candidates(), ", "))); " *
              "set CUDA_PATH to point to one")
    end
    return dir
end

function toolkit_candidates()
    candidates = String[]
    for var in ("CUDA_PATH", "CUDA_HOME")
        haskey(ENV, var) && push!(candidates, ENV[var])
    end
    append!(candidates, ["/usr/local/cuda", "/opt/cuda"])
    return candidates
end

function find_toolkit()
    for dir in toolkit_candidates()
        if isfile(joinpath(dir, "include", "cuda_runtime.h")) &&
           isfile(joinpath(dir, "nvvm", "libdevice", "libdevice.10.bc"))
            return dir
        end
    end
    return nothing
end

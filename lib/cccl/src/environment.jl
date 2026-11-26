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

const toolkit = Ref{Union{Nothing,String}}(nothing)

"""
    CCCL.toolkit_path() -> String

The CUDA toolkit whose headers, `libdevice` and CUDA runtime library HostJIT compiles and
links the algorithms against.

When CUDA.jl uses the CUDA JLLs, this is a directory assembled from those JLLs, as long as
they ship every header HostJIT needs. Otherwise, it is the local toolkit CUDA.jl uses, or
the one `CUDA_PATH` or `CUDA_HOME` points to, or one in a standard location.
"""
function toolkit_path()
    dir = toolkit[]
    if dir === nothing
        error("CCCL needs a CUDA toolkit, but none was found (looked in $(join(toolkit_candidates(), ", "))); " *
              "set CUDA_PATH to point to one")
    end
    return dir
end

function toolkit_candidates()
    candidates = String[]

    # the local toolkit CUDA.jl uses, if any
    compiler = CUDACore.CUDA_Compiler
    if CUDACore.local_compiler && isdefined(compiler, :libdevice)
        # <toolkit>/nvvm/libdevice/libdevice.10.bc
        push!(candidates, dirname(dirname(dirname(compiler.libdevice))))
    end

    for var in ("CUDA_PATH", "CUDA_HOME")
        haskey(ENV, var) && push!(candidates, ENV[var])
    end
    append!(candidates, ["/usr/local/cuda", "/opt/cuda"])
    return unique!(candidates)
end

function find_toolkit()
    compiler, runtime = CUDACore.CUDA_Compiler, CUDACore.CUDA_Runtime
    if isdefined(compiler, :cuda_include_dir) && isdefined(runtime, :libcudart)
        dir = assemble_toolkit(compiler.cuda_include_dir, compiler.libdevice,
                               runtime.libcudart)
        dir === nothing || return dir
    end

    for dir in toolkit_candidates()
        if isfile(joinpath(dir, "include", "cuda_runtime.h")) &&
           isfile(joinpath(dir, "nvvm", "libdevice", "libdevice.10.bc"))
            return dir
        end
    end
    return nothing
end

# HostJIT expects a toolkit laid out like a local installation: `include` with the CUDA
# runtime headers, including the compiler-support headers in `include/crt`,
# `nvvm/libdevice/libdevice.10.bc`, and the CUDA runtime library in `lib64`, which it
# links the generated libraries against and adds to their RPATH. The JLLs provide these
# pieces in separate artifacts, so link them together in a scratch directory.
function assemble_toolkit(include_dir, libdevice, libcudart)
    isfile(joinpath(include_dir, "crt", "host_defines.h")) || return nothing

    sources = (include_dir, libdevice, libcudart)
    dir = joinpath(@get_scratch!("toolkits"), string(hash(sources); base=62))
    isdir(dir) && return dir

    # assemble the toolkit in a temporary directory and rename it into place, so that
    # concurrent sessions never see a partial one
    temp = mktempdir(dirname(dir); cleanup=false)
    try
        symlink(include_dir, joinpath(temp, "include"))

        mkpath(joinpath(temp, "nvvm", "libdevice"))
        symlink(libdevice, joinpath(temp, "nvvm", "libdevice", "libdevice.10.bc"))

        # the generated libraries load the runtime by its soname
        mkpath(joinpath(temp, "lib64"))
        libdir = dirname(realpath(libcudart))
        for name in readdir(libdir)
            startswith(name, "libcudart.so") || continue
            symlink(joinpath(libdir, name), joinpath(temp, "lib64", name))
        end

        mv(temp, dir)
    catch
        rm(temp; recursive=true, force=true)
        isdir(dir) || rethrow()
    end
    return dir
end

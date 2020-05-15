export has_cuda, has_cuda_gpu, usable_cuda_gpus

"""
    has_cuda()::Bool

Check whether the local system provides an installation of the CUDA driver and toolkit.
Use this function if your code loads packages that require CUDA, such as CuArrays.jl.

Note that CUDA-dependent packages might still fail to load if the installation is broken,
so it's recommended to guard against that and print a warning to inform the user:

```
using CUDAapi
if has_cuda()
    try
        using CuArrays
    catch ex
        @warn "CUDA is installed, but CuArrays.jl fails to load" exception=(ex,catch_backtrace())
    end
end
```
"""
function has_cuda()
    toolkit_dirs = find_toolkit()

    # check for the CUDA driver library
    libcuda = find_cuda_library("cuda", toolkit_dirs)
    return libcuda !== nothing
end

"""
    has_cuda_gpu()::Bool

Check whether the local system provides an installation of the CUDA driver and toolkit, and
if it contains a CUDA-capable GPU. See [`has_cuda`](@ref) for more details.

Note that this function initializes the CUDA API in order to check for the number of GPUs.
"""
function has_cuda_gpu()
    toolkit_dirs = find_toolkit()

    # find the CUDA driver library
    libcuda = find_cuda_library("cuda", toolkit_dirs)
    if libcuda === nothing
        return false
    end
    lib = Libdl.dlopen(libcuda)

    # initialize the API
    cuInit = Libdl.dlsym(lib, :cuInit)
    status = ccall(cuInit, Cint, (Cuint,), 0)
    if status != 0
        @warn "CUDA is installed, but fails to load (error code $status)"
        return false
    end

    # count the GPUs
    cuDeviceGetCount = Libdl.dlsym(lib, :cuDeviceGetCount)
    count_ref = Ref{Cint}()
    status = ccall(cuDeviceGetCount, Cint, (Ptr{Cint},), count_ref)
    @assert status == 0
    return count_ref[] > 0
end

"""
    usable_cuda_gpus(; suppress_output=false)::Int

Returns the number of CUDA GPUs that are available for use on the local system.

Note that this function initializes the CUDA API in order to check for the number of GPUs.
"""
function usable_cuda_gpus(; suppress_output=false)
    toolkit_dirs = find_toolkit()

    # find the CUDA driver library
    libcuda = find_cuda_library("cuda", toolkit_dirs)
    if libcuda === nothing
        return 0
    end
    lib = Libdl.dlopen(libcuda)

    # initialize the API
    cuInit = Libdl.dlsym(lib, :cuInit)
    status = ccall(cuInit, Cint, (Cuint,), 0)
    if status != 0
        !suppress_output && @warn "CUDA is installed, but fails to load (error code $status)"
        return -1
    end

    # count the GPUs
    cuDeviceGetCount = Libdl.dlsym(lib, :cuDeviceGetCount)
    count_ref = Ref{Cint}()
    status = ccall(cuDeviceGetCount, Cint, (Ptr{Cint},), count_ref)
    @assert status == 0
    return Int(count_ref[])
end

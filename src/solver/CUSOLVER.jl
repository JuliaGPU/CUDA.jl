module CUSOLVER

import CUDAdrv
using CUDAdrv: CuContext, CuDevice
using CUDAnative

using ..CuArrays
const cudaStream_t = Ptr{Nothing}

using ..CuArrays: libcusolver, configured, _getindex

using LinearAlgebra

import Base.one
import Base.zero

include("libcusolver_types.jl")
include("error.jl")
include("libcusolver.jl")

const libcusolver_handles_dense = Dict{CuContext,cusolverDnHandle_t}()
const libcusolver_handle_dense = Ref{cusolverDnHandle_t}()

function __init__()
    configured || return

    # initialize the library when we switch devices
    callback = (dev::CuDevice, ctx::CuContext) -> begin
        libcusolver_handle_dense[] = get!(libcusolver_handles_dense, ctx) do
            @debug "Initializing CUSOLVER for $dev"
            handle = Ref{cusolverDnHandle_t}()
            cusolverDnCreate(handle)
            handle[]
        end
    end
    push!(CUDAnative.device!_listeners, callback)

    # deinitialize when exiting
    atexit() do
        libcusolver_handle_dense[] = C_NULL

        for (ctx, handle) in libcusolver_handles_dense
            if CUDAdrv.isvalid(ctx)
                cusolverDnDestroy(handle)
            end
        end
    end
end

include("dense.jl")
include("highlevel.jl")

end

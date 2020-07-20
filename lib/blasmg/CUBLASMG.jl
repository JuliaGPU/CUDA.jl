module CUBLASMG

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using ..CuArrays
using ..CuArrays: libcublasmg, libcudalibmg, unsafe_free!, @retry_reclaim
using ..CuArrays.CUDALIBMG: CudaLibMGDescriptor, cudaLibMgGetLocalMatrixDimensions, cudaLibMgCreateDeviceGrid, cudaLibMgMatrixDesc_t, cudaLibMgGrid_t, CudaLibMGGrid, CUDALIBMG_GRID_MAPPING_COL_MAJOR, CUDALIBMG_GRID_MAPPING_ROW_MAJOR 
using ..CuArrays.CUBLAS: cublasStatus_t, cublasop, cublasOperation_t, CUBLAS_STATUS_ALLOC_FAILED, CUBLAS_STATUS_SUCCESS, CUBLASError
using LinearAlgebra

using CEnum

const cudaDataType_t = cudaDataType

# core library
include("libcublasmg_common.jl")
include("error.jl")
include("libcublasmg.jl")

# low-level wrappers
include("wrappers.jl")

# high-level integrations

# thread cache for task-local library handles
const thread_handles = Vector{Union{Nothing,cublasMgHandle_t}}()

function mg_handle()
    tid = Threads.threadid()
    if @inbounds thread_handles[tid] === nothing
        ctx = context()
        thread_handles[tid] = get!(task_local_storage(), (:CUBLASMG, ctx)) do
            handle = cublasMgCreate()
            finalizer(current_task()) do task
                CUDAdrv.isvalid(ctx) || return
                context!(ctx) do
                    cublasMgDestroy(handle)
                end
            end
            handle
        end
    end
    @inbounds thread_handles[tid]
end

function __init__()
    resize!(thread_handles, Threads.nthreads())
    fill!(thread_handles, nothing)

    CUDAnative.atcontextswitch() do tid, ctx
        thread_handles[tid] = nothing
    end

    CUDAnative.attaskswitch() do tid, task
        thread_handles[tid] = nothing
    end
end

end

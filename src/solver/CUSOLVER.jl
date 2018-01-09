module CUSOLVER

using ..CuArrays
const cudaStream_t = Ptr{Void}

using ..CuArrays: libcusolver, configured, _getindex

import Base.one
import Base.zero

include("libcusolver_types.jl")
include("error.jl")
include("libcusolver.jl")

const libcusolver_handle_dense = Ref{cusolverDnHandle_t}()

function __init__()
  configured || return

  cusolverDnCreate(libcusolver_handle_dense)
  atexit(() -> cusolverDnDestroy(libcusolver_handle_dense[]))
end

include("dense.jl")
include("highlevel.jl")

end

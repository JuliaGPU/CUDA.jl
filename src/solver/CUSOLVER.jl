module CUSOLVER

using ..CuArrays
const cudaStream_t = Ptr{Void}

using ..CuArrays: libcusolver, configured, _getindex

import Base.one
import Base.zero

include("libcusolver_types.jl")
include("error.jl")
include("libcusolver.jl")

const cusolverDnhandle = cusolverDnHandle_t[0]

function __init__()
  configured || return

  cusolverDnCreate(cusolverDnhandle)
  atexit(() -> cusolverDnDestroy(cusolverDnhandle[1]))
end

include("dense.jl")
include("highlevel.jl")

end

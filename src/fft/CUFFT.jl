module CUFFT

using CUDAapi

using ..CuArrays
import ..CuArrays: unsafe_free!

using CUDAdrv
using CUDAdrv: CUstream

using CEnum

const libcufft = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "cufft64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("cufft", toolkit)
    if path === nothing
        error("Could not find libcufft")
    end
    basename(path)
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcufft"
end

# core library
include("libcufft_common.jl")
include("error.jl")
include("libcufft.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("fft.jl")

end

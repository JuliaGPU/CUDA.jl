# Device intrinsics, corresponding with CUDA extensions to the C language

# TODO: "CUDA C programming guide" > "C language extensions" lists mathematical functions,
#       without mentioning libdevice. Is this implied, by NVCC always using libdevice,
#       or are there some natively-supported math functions as well?

for i in ["memory_shared", "indexing", "synchronization",
          "warp_vote", "warp_shuffle", "output"]
    include(joinpath("intrinsics", "$i.jl"))
end

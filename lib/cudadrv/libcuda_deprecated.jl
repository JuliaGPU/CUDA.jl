# wrappers for old functions that aren't in the headers anymore

# TODO: auto-generate these as soon as we detect a versioned symbol?


## superseded in CUDA 11

@checked function cuDevicePrimaryCtxRelease(dev)
    @runtime_ccall((:cuDevicePrimaryCtxRelease, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuDevicePrimaryCtxSetFlags(dev, flags)
    @runtime_ccall((:cuDevicePrimaryCtxSetFlags, libcuda()), CUresult,
                   (CUdevice, UInt32),
                   dev, flags)
end

@checked function cuDevicePrimaryCtxReset(dev)
    @runtime_ccall((:cuDevicePrimaryCtxReset, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer,
                                        bufferSize)
    initialize_api()
    @runtime_ccall((:cuGraphInstantiate, libcuda()), CUresult,
                   (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t),
                   phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end

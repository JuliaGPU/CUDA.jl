## superseded in CUDA 11

@checked function cuDevicePrimaryCtxRelease(dev)
    ccall((:cuDevicePrimaryCtxRelease, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuDevicePrimaryCtxSetFlags(dev, flags)
    ccall((:cuDevicePrimaryCtxSetFlags, libcuda()), CUresult,
                   (CUdevice, UInt32),
                   dev, flags)
end

@checked function cuDevicePrimaryCtxReset(dev)
    ccall((:cuDevicePrimaryCtxReset, libcuda()), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer,
                                        bufferSize)
    initialize_api()
    ccall((:cuGraphInstantiate, libcuda()), CUresult,
                   (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t),
                   phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end

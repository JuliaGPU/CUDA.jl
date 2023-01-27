## superseded in CUDA 11.0

@checked function cuDevicePrimaryCtxRelease(dev)
    ccall((:cuDevicePrimaryCtxRelease, libcuda), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuDevicePrimaryCtxSetFlags(dev, flags)
    ccall((:cuDevicePrimaryCtxSetFlags, libcuda), CUresult,
                   (CUdevice, UInt32),
                   dev, flags)
end

@checked function cuDevicePrimaryCtxReset(dev)
    ccall((:cuDevicePrimaryCtxReset, libcuda), CUresult,
                   (CUdevice,),
                   dev)
end

@checked function cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer,
                                        bufferSize)
    ccall((:cuGraphInstantiate, libcuda), CUresult,
                   (Ptr{CUgraphExec}, CUgraph, Ptr{CUgraphNode}, Cstring, Csize_t),
                   phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
end


## deprecated in CUDA 12.0

@checked function cuGraphExecUpdate(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    initialize_context()
    @ccall libcuda.cuGraphExecUpdate(hGraphExec::CUgraphExec, hGraph::CUgraph,
                                     hErrorNode_out::Ptr{CUgraphNode},
                                     updateResult_out::Ptr{CUgraphExecUpdateResult})::CUresult
end

@checked function cuGraphInstantiate_v2(phGraphExec, hGraph, phErrorNode, logBuffer,
                                        bufferSize)
    initialize_context()
    @ccall libcuda.cuGraphInstantiate_v2(phGraphExec::Ptr{CUgraphExec}, hGraph::CUgraph,
                                         phErrorNode::Ptr{CUgraphNode}, logBuffer::Cstring,
                                         bufferSize::Csize_t)::CUresult
end

@checked function cuStreamGetCaptureInfo(hStream, captureStatus_out, id_out)
    initialize_context()
    @ccall libcuda.cuStreamGetCaptureInfo(hStream::CUstream,
                                          captureStatus_out::Ptr{CUstreamCaptureStatus},
                                          id_out::Ptr{cuuint64_t})::CUresult
end

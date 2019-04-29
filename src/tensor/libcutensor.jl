# Julia wrapper for header: /usr/local/cuda/include/cusparse.h

#helper functions
function cutensorCreate()
  handle = Ref{cutensorHandle_t}()
  @check ccall((:cutensorCreate, libcutensor), cutensorStatus_t, (Ptr{cutensorHandle_t},), handle)
  handle[]
end
function cutensorDestroy(handle)
  @check ccall((:cutensorDestroy, libcutensor), cutensorStatus_t, (cutensorHandle_t,), handle)
end
function cutensorGetVersion(handle, version)
  @check ccall((:cutensorGetVersion, libcutensor), cutensorStatus_t, (cutensorHandle_t, Ptr{Cint}), handle, version)
end
function cutensorCreateTensorDescriptor(numModes::Cint, extent::Vector{Int64}, stride::Vector{Int64}, T::cudaDataType_t, unaryOp::cutensorOperator_t, vectorWidth::Cint, vectorModeIndex::Cint)
  desc = Ref{cutensorTensorDescriptor_t}(C_NULL)
  @check ccall((:cutensorCreateTensorDescriptor, libcutensor), cutensorStatus_t, (Ref{cutensorTensorDescriptor_t}, Cint, Ptr{Int64}, Ptr{Int64}, cudaDataType_t, cutensorOperator_t, Cint, Cint), desc, numModes, extent, stride, T, unaryOp, vectorWidth, vectorModeIndex)
  return desc[]
end
function cutensorDestroyTensorDescriptor(desc::cutensorTensorDescriptor_t)
  @check ccall((:cutensorDestroyTensorDescriptor, libcutensor), cutensorStatus_t, (cutensorTensorDescriptor_t,), desc)
end

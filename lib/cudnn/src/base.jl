function cudnnCreate()
    handle_ref = Ref{cudnnHandle_t}()
    cudnnCreate(handle_ref)
    return handle_ref[]
end

function cudnnGetProperty(property::CUDACore.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end

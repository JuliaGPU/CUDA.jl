using CUDAdrv

function alloc(bytes)
  try
    Mem.alloc(bytes)
  catch e
    e == CUDAdrv.CuError(2) || rethrow()
    gc(false)
    Mem.alloc(bytes)
  end
end

function dealloc(ptr)
  CUDAdrv.isvalid(ptr.ctx) && Mem.free(ptr)
end

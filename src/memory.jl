function alloc(bytes)
  Mem.alloc(bytes)
end

function dealloc(ptr)
  CUDAdrv.isvalid(ptr.ctx) && Mem.free(ptr)
end

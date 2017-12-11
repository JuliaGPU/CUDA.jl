using CUDAdrv

const pools = Vector{Mem.Buffer}[]

poolidx(n) = ceil(Int, log2(n))+1
poolsize(idx) = 2^(idx-1)

function pool(idx)
  while length(pools) < idx
    push!(pools, Mem.Buffer[])
  end
  @inbounds return pools[idx]
end

function clearpool()
  for pool in pools
    for buf in pool
      Mem.free(buf)
    end
    empty!(pool)
  end
end

function alloc(bytes)
  pid = poolidx(bytes)
  pl = pool(pid)
  isempty(pl) || return pop!(pl)
  gc(false)
  isempty(pl) || return pop!(pl)
  try
    Mem.alloc(poolsize(pid))
  catch e
    e == CUDAdrv.CuError(2) || rethrow()
    clearpool()
    gc()
    isempty(pl) || return pop!(pl)
    clearpool()
    Mem.alloc(poolsize(pid))
  end
end

function dealloc(buf, n)
  push!(pool(poolidx(n)), buf)
end

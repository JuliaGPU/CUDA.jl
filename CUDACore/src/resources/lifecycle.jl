## retirement of resources freed by finalizers
#
# finalizers run on whatever thread triggers a collection, or releases a lock while
# finalizers are pending, and cannot switch tasks. releasing memory there could block that
# thread, as some driver calls wait for unrelated GPU work to finish (also blocking kernel
# launches from other threads in the meantime). if that work depends on a task on this
# thread, that would deadlock. so finalizers only retire resources, pushing them onto a
# lock-free list, and regular tasks release them.

mutable struct Retired
  const resource::Any
  next::Union{Nothing,Retired}
end

mutable struct RetiredList
  Base.@atomic head::Union{Nothing,Retired}
end
const retired_memory = RetiredList(nothing)

function push_retired!(first::Retired, last::Retired)
  head = Base.@atomic :monotonic retired_memory.head
  while true
    last.next = head
    head, ok = Base.@atomicreplace :release :monotonic retired_memory.head head => first
    ok && return
  end
end

# can be called from a finalizer: doesn't switch tasks, take locks or call into CUDA.
# `release_now(resource)` is called later on, by a regular task.
function retire!(resource)
  node = Retired(resource, nothing)
  push_retired!(node, node)
  return
end


## exclusion of graph captures
#
# while a stream is being captured in global mode, other threads may not make API calls that
# could interfere with the capture, as releasing resources does. so releasing resources and
# starting a capture exclude each other, but both only hold `capture_lock` briefly: GPU work
# and other allocations can proceed while resources are being released. a capture that
# starts in the meantime waits for that to finish, unless it involves operations that may wait
# for the GPU (like reclaiming memory does), for which the capture fails instead of waiting an
# unbounded amount of time.

const capture_lock = ReentrantLock()
const active_releases = Ref(0)            # protected by capture_lock
const active_blocking_releases = Ref(0)   # protected by capture_lock

function begin_capture()
  while true
    @lock capture_lock begin
      if active_releases[] == 0
        Threads.atomic_add!(active_captures, 1)
        return
      end
      if active_blocking_releases[] > 0
        error("Cannot start a graph capture while CUDA.jl is waiting for the GPU to " *
              "release resources, e.g., during `CUDA.reclaim()`.")
      end
    end
    yield()
  end
end
end_capture() = (Threads.atomic_sub!(active_captures, 1); nothing)

# run `f`, which releases resources, unless a capture is in progress. returns whether it ran.
function releasing(f; blocking::Bool=false)
  GC.in_finalizer() && return false
  @lock capture_lock begin
    active_captures[] == 0 || return false
    active_releases[] += 1
    blocking && (active_blocking_releases[] += 1)
  end
  try
    f()
  finally
    @lock capture_lock begin
      active_releases[] -= 1
      blocking && (active_blocking_releases[] -= 1)
    end
  end
  return true
end

# release a resource right away if possible, or retire it otherwise
@inline function discard(resource)
  if GC.in_finalizer() || active_captures[] > 0 || !releasing(() -> release_now(resource))
    retire!(resource)
  end
  return
end


## deferred cleanup of arbitrary objects

struct ReleaseAction{F,T}
  f::F
  obj::T
  ctx::Union{Nothing,CuContext}
  blocking::Bool
end

@public resource_finalizer

"""
    resource_finalizer(f, obj; blocking=true, ctx=context()) -> obj

Register `f` to clean up `obj` once it is garbage collected, like `finalizer(f, obj)`, but
without calling `f` from the garbage collector. Many CUDA API calls that release resources
wait for all running kernels to finish, which can stall or deadlock the thread the garbage
collector happens to run on. Instead, the finalizer only queues `obj`, and a regular task
calls `f(obj)` later on, in the context `ctx` that was active when registering. Pass
`ctx=nothing` if `f` activates the right context itself.

With `blocking=true` (the default), `f` is only called when CUDA.jl reclaims memory, i.e.,
when an allocation runs out of memory, or when calling `CUDA.reclaim()`. Only use
`blocking=false` when `f` is known not to wait for the GPU. It is then called the next time
CUDA.jl allocates memory or synchronizes, or within a second.

The cleanup function should use its argument instead of capturing `obj`, which would keep
it alive. If `f` throws an error, it is logged, and `obj` is kept alive instead of retrying
the cleanup, as it may have partially succeeded. As with `finalizer`, calling
`finalize(obj)` only queues the cleanup, and there is no guarantee that `f` is called before
the process exits. The callers remain responsible for keeping `obj` alive while the GPU uses
it, and for not cleaning it up twice when also releasing it explicitly.
"""
function resource_finalizer(f, obj; blocking::Bool=true,
                            ctx::Union{Nothing,CuContext}=context())
  finalizer(obj) do obj
    retire!(ReleaseAction(f, obj, ctx, blocking))
  end
  return obj
end

# for CUDACore's own types, which implement `release_now` (it must not wait for the GPU)
function resource_finalizer(obj)
  finalizer(retire!, obj)
  return obj
end

function defer_release(f, obj; blocking=false, ctx=nothing)
  discard(ReleaseAction(f, obj, ctx, blocking))
  return
end

destroy_later(f, obj) = defer_release(f, obj; blocking=true)

function run_release(action::ReleaseAction)
  if action.ctx === nothing
    Base.invokelatest(action.f, action.obj)
  else
    context!(action.ctx) do
      Base.invokelatest(action.f, action.obj)
    end
  end
  return
end


## held resources
#
# resources that can only be released once the GPU has finished using them (as indicated
# by an event), or whose release may wait for the GPU (so is deferred until memory is
# reclaimed). this also keeps resources whose release failed: as that may have partially
# succeeded, they are kept alive instead of retrying.

struct HeldResource
  action::ReleaseAction
  gate::Union{Nothing,CuEvent}  # nothing: release when reclaiming memory
  failed::Bool
end
struct ResourceHolds <: Reclaimable
  lock::ReentrantLock
  # partitioned so that polling for completion does not have to scan the resources that
  # are only released when reclaiming memory
  completion::Vector{HeldResource}
  reclaim::Vector{HeldResource}
  count::Threads.Atomic{Int}  # length(completion), for checking without the lock
end
const resource_holds = ResourceHolds(ReentrantLock(), HeldResource[], HeldResource[],
                                     Threads.Atomic{Int}(0))

hold!(action::ReleaseAction, gate=nothing; failed=false) =
  hold!(HeldResource(action, gate, failed))
function hold!(held::HeldResource)
  @lock resource_holds.lock begin
    completion = held.gate !== nothing && !held.failed
    push!(completion ? resource_holds.completion : resource_holds.reclaim, held)
    completion && Threads.atomic_add!(resource_holds.count, 1)
  end
  return
end

function release_failed!(action::ReleaseAction, err, bt)
  hold!(action; failed=true)
  @error "Failed to release a GPU resource; keeping it alive" exception=(err, bt)
  return
end

# returns whether the release succeeded; if not, the resource is kept alive
function attempt_release(action::ReleaseAction)
  try
    run_release(action)
    return true
  catch err
    release_failed!(action, err, catch_backtrace())
    return false
  end
end

function release_now(action::ReleaseAction)
  action.blocking ? hold!(action) : attempt_release(action)
  return
end

# release held resources whose GPU work has finished, or with `reclaim=true`, also those
# whose release may wait for the GPU. `select` makes it possible to only release some of
# them. resources are detached first, as releasing them may hold new ones.
function release_held!(; reclaim=false, select=Returns(true))
  !reclaim && resource_holds.count[] == 0 && return
  items = @lock resource_holds.lock begin
    items = copy(resource_holds.completion)
    empty!(resource_holds.completion)
    resource_holds.count[] = 0
    if reclaim
      append!(items, resource_holds.reclaim)
      empty!(resource_holds.reclaim)
    end
    items
  end
  for held in items
    action, gate = held.action, held.gate
    if held.failed || !select(action)
      hold!(held)
      continue
    end
    try
      if gate !== nothing && !isdone(gate)
        hold!(held)
        continue
      end
      run_release(action)
    catch err
      release_failed!(action, err, catch_backtrace())
    end
  end
  return
end
purge!(::ResourceHolds) = (drain_retired(); release_held!(; reclaim=true))


## draining retired resources

# retired resources are released by one task at a time, holding this lock. it is never held
# while waiting for the GPU. a bounded drain leaves the rest for a later one (in
# `retired_backlog`), without having to put it back.
const drain_lock = ReentrantLock()
const retired_backlog = RetiredList(nothing)
const drain_running = Ref(false) # protected by drain_lock

"""
    drain_retired([limit])

Release (at most `limit`) resources that have been retired by finalizers, e.g., making
memory available to future allocations. Only resources retired before the call are
considered. Returns the number of resources released.
"""
function drain_retired(limit::Int=typemax(Int))
  (GC.in_finalizer() || active_captures[] > 0) && return 0
  if (Base.@atomic :monotonic retired_memory.head) === nothing &&
     (Base.@atomic :monotonic retired_backlog.head) === nothing
    resource_holds.count[] == 0 && return 0
  end

  # exhaustive drains wait for others to finish, bounded ones (as used when allocating)
  # leave the work to them
  if limit == typemax(Int)
    lock(drain_lock)
  else
    trylock(drain_lock) || return 0
  end
  n = 0
  try
    # releasing a resource may drain recursively, which should leave the work to the
    # outer drain
    drain_running[] && return 0
    drain_running[] = true
    try
      releasing() do
        if (Base.@atomic :monotonic retired_backlog.head) === nothing
          Base.@atomic :monotonic retired_backlog.head =
            Base.@atomicswap :acquire retired_memory.head = nothing
        end
        while n < limit
          node = Base.@atomic :monotonic retired_backlog.head
          node === nothing && break
          Base.@atomic :monotonic retired_backlog.head = node.next
          try
            release_now(node.resource)
          catch err
            release_failed!(ReleaseAction(identity, node.resource, nothing, false),
                            err, catch_backtrace())
          end
          n += 1
        end
        release_held!()
      end
    finally
      drain_running[] = false
    end
  finally
    unlock(drain_lock)
  end
  return n
end

# drain periodically, so that memory gets released when the application stops using CUDA.
# started when a context is first used (see `disposal_stream`).
const retired_drainer = Threads.Atomic{Bool}(false)
function start_retired_drainer()
  retired_drainer[] && return
  Threads.atomic_cas!(retired_drainer, false, true) && return
  # an active timer would keep precompilation from finishing
  ccall(:jl_generating_output, Cint, ()) == 0 || return
  Timer(1; interval=1) do _
    try
      drain_retired()
    catch err
      @error "Failed to release retired GPU resources" exception=(err, catch_backtrace())
    end
  end
  return
end

# how many retired resources to release when allocating, to bound allocation latency
const ALLOC_DRAIN_LIMIT = 256

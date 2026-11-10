## segments and blocks
#
# the bookkeeping of the caching allocator (see caching.jl). it doesn't depend on CUDA, so
# that its invariants can be tested against synthetic fences (see test/core/segments.jl).
#
# backing memory is allocated in segments, which are split into blocks. small, medium and
# large requests are served from separate classes of segments, so that small blocks don't
# pin large segments. a block that is freed may still be in use by the GPU, so it carries a
# fence: an event recorded after its last use, on the stream it was last used on (its lane).
# the block can be reused right away on that lane, as work submitted to a stream runs after
# the work submitted to it before, but on other streams only once the fence has completed.
# no waits are inserted to reuse memory earlier.
#
# segments are only returned once all of their blocks are free. that happens in stream
# order after their fences, so it doesn't wait for the GPU either.
#
# nothing here is thread-safe: the caller serializes all operations on an arena.

const SEGMENT_ALIGN = 512
const SMALL_REQUEST = 1 << 20   # requests up to this size
const SMALL_SEGMENT = 2 << 20   # are suballocated from segments of this size,
const LARGE_REQUEST = 10 << 20  # requests smaller than this
const MEDIUM_SEGMENT = 20 << 20 # from segments of this size,
const LARGE_ROUND = 2 << 20     # and larger ones get their own segment, rounded to this.
const OVERSIZE = 64 << 20       # blocks this large are only used for requests within an
                                # eighth of their size, and aren't split

@enum SegmentClass::UInt8 SMALL_SEGMENTS MEDIUM_SEGMENTS LARGE_SEGMENTS

block_capacity(sz::Integer) = cld(sz, SEGMENT_ALIGN) * SEGMENT_ALIGN
segment_class(cap::Integer) = cap <= SMALL_REQUEST ? SMALL_SEGMENTS :
                              cap < LARGE_REQUEST ? MEDIUM_SEGMENTS : LARGE_SEGMENTS
function segment_size(cap::Integer)
  class = segment_class(cap)
  class == SMALL_SEGMENTS ? SMALL_SEGMENT :
  class == MEDIUM_SEGMENTS ? MEDIUM_SEGMENT : cld(cap, LARGE_ROUND) * LARGE_ROUND
end
# the smallest remainder that is worth splitting off a block
split_threshold(class::SegmentClass) =
  class == SMALL_SEGMENTS ? SEGMENT_ALIGN :
  class == MEDIUM_SEGMENTS ? SMALL_REQUEST + 1 : LARGE_REQUEST

@enum SegmentState::UInt8 begin
  RESIDENT          # its blocks are allocated and reused
  RELEASE_QUEUED    # being returned to the driver (never reused again)
  RELEASED
  RELEASE_FAILED    # kept, as the release may have partially succeeded
end

mutable struct Segment
  const base::UInt
  const size::Int
  const class::SegmentClass
  state::SegmentState
  head::Any                 # the block at the start of the segment
  nlive::Int                # number of live blocks
  free_since::UInt64        # when it last became wholly free (in ns)
  free_slot::Int            # its index in the arena's wholly free segments, or 0
  payload::Any              # for the back-end (e.g., a sanitizer annotation)
end

# an event recorded after the last use of free blocks. its lane identifies the stream the
# memory was last used on, whose later work is ordered after that use. `recorder` identifies
# the stream it was recorded on, and `seq` orders it among the fences recorded on that
# stream, so a later fence covers an earlier one. that can be a different stream than the
# lane's, e.g., when the lane's stream has been destroyed.
mutable struct Fence{E,L,R}
  const event::E
  const lane::L
  const recorder::R
  const seq::UInt64
  # the free blocks it guards (a list through `Block.fence_next`)
  blocks::Any
  nblocks::Int
  # whether it's still waiting to be found completed, in its recorder's queue
  queued::Bool
  prev::Union{Nothing,Fence{E,L,R}}
  next::Union{Nothing,Fence{E,L,R}}
  holds::Int                # evictions that still need to be ordered after it
  done::Bool
end
Fence(event::E, lane::L, recorder::R, seq::Integer) where {E,L,R} =
  Fence{E,L,R}(event, lane, recorder, seq, nothing, 0, true, nothing, nothing, 0, false)

# the fences recorded on a stream that haven't been found completed, in recording order
mutable struct FenceQueue{F}
  head::Union{Nothing,F}
  tail::Union{Nothing,F}
end

# whether all uses guarded by `b` are also guarded by `a`
covers(a::Fence, b::Fence) = a === b || (isequal(a.recorder, b.recorder) && a.seq > b.seq)

@enum BlockIndex::UInt8 NOT_INDEXED COMPLETED_INDEX LANE_INDEX

mutable struct Block{F}
  const segment::Segment
  const addr::UInt
  size::Int                 # capacity
  requested::Int            # bytes requested by a live allocation
  live::Bool
  fence::Union{Nothing,F}   # for a free block, nothing when its last use has completed
  index::BlockIndex
  prev::Union{Nothing,Block{F}}
  next::Union{Nothing,Block{F}}
  # the other blocks guarded by the same fence
  fence_prev::Union{Nothing,Block{F}}
  fence_next::Union{Nothing,Block{F}}
  # treap node
  left::Union{Nothing,Block{F}}
  right::Union{Nothing,Block{F}}
  priority::UInt64
end
Block{F}(segment::Segment, addr::UInt, size::Int, priority::UInt64) where {F} =
  Block{F}(segment, addr, size, 0, false, nothing, NOT_INDEXED, nothing, nothing, nothing,
           nothing, nothing, nothing, priority)

function link_block!(f::Fence, b::Block)
  head = f.blocks
  b.fence_prev = nothing
  b.fence_next = head
  head === nothing || (head.fence_prev = b)
  f.blocks = b
  f.nblocks += 1
  return
end
function unlink_block!(f::Fence, b::Block)
  p, n = b.fence_prev, b.fence_next
  p === nothing ? (f.blocks = n) : (p.fence_next = n)
  n === nothing || (n.fence_prev = p)
  b.fence_prev = b.fence_next = nothing
  f.nblocks -= 1
  return
end
function fence_blocks(f::Fence{E,L,R}) where {E,L,R}
  B = Block{Fence{E,L,R}}
  blocks = B[]
  b = f.blocks
  while b !== nothing
    push!(blocks, b::B)
    b = b.fence_next
  end
  return blocks
end


## free block index
#
# a treap keyed by (size, address), so that the best fitting block is found in logarithmic
# time (expected). the blocks themselves are the nodes.

mutable struct BlockTree{B}
  root::Union{Nothing,B}
end
BlockTree{B}() where {B} = BlockTree{B}(nothing)

Base.isempty(t::BlockTree) = t.root === nothing

before(a::Block, b::Block) = a.size < b.size || (a.size == b.size && a.addr < b.addr)

function Base.push!(t::BlockTree{B}, x::B) where {B}
  x.left = x.right = nothing
  t.root = tree_insert(t.root, x)
  return t
end
function tree_insert(node::Union{Nothing,B}, x::B) where {B}
  node === nothing && return x
  if x.priority > node.priority
    x.left, x.right = tree_split(node, x)
    return x
  end
  if before(x, node)
    node.left = tree_insert(node.left, x)
  else
    node.right = tree_insert(node.right, x)
  end
  return node
end
# split a tree into the nodes before `x`, and those after it
function tree_split(node::Union{Nothing,B}, x::B) where {B}
  node === nothing && return (nothing, nothing)
  if before(node, x)
    l, r = tree_split(node.right, x)
    node.right = l
    return (node, r)
  else
    l, r = tree_split(node.left, x)
    node.left = r
    return (l, node)
  end
end

function Base.delete!(t::BlockTree{B}, x::B) where {B}
  t.root = tree_delete(t.root, x)
  x.left = x.right = nothing
  return t
end
function tree_delete(node::Union{Nothing,B}, x::B) where {B}
  node === nothing && error("block is not in the index")
  node === x && return tree_merge(node.left, node.right)
  if before(x, node)
    node.left = tree_delete(node.left, x)
  else
    node.right = tree_delete(node.right, x)
  end
  return node
end
# merge two trees whose nodes are all ordered before those of the other one
function tree_merge(a::Union{Nothing,B}, b::Union{Nothing,B}) where {B}
  a === nothing && return b
  b === nothing && return a
  if a.priority > b.priority
    a.right = tree_merge(a.right, b)
    return a
  else
    b.left = tree_merge(a, b.left)
    return b
  end
end

# the smallest block that has at least `sz` bytes
function smallest_fit(t::BlockTree, sz::Int)
  best = nothing
  node = t.root
  while node !== nothing
    if node.size >= sz
      best = node
      node = node.left
    else
      node = node.right
    end
  end
  return best
end

Base.foreach(f, t::BlockTree) = (tree_foreach(f, t.root); nothing)
function tree_foreach(f, node)
  node === nothing && return
  tree_foreach(f, node.left)
  f(node)
  tree_foreach(f, node.right)
end


## arena
#
# the segments allocated in one context, with indexes of their free blocks per class: one
# for blocks whose fences have completed, and one per lane for blocks still waiting for
# theirs. fences are polled in the order they were recorded per stream, which is the order
# they complete in.

const BlockTrees{B} = NTuple{3,BlockTree{B}}
block_trees(::Type{B}) where {B} = (BlockTree{B}(), BlockTree{B}(), BlockTree{B}())

mutable struct Arena{E,L,R}
  const completed::BlockTrees{Block{Fence{E,L,R}}}
  const lanes::Dict{L,BlockTrees{Block{Fence{E,L,R}}}}
  const fences::Dict{R,FenceQueue{Fence{E,L,R}}}
  const live::Dict{UInt,Block{Fence{E,L,R}}}
  const segments::Base.IdSet{Segment}     # resident ones
  const wholly_free::Vector{Segment}      # resident ones without live blocks
  # events of fences that aren't needed anymore, for the back-end to reuse
  const spare_events::Vector{E}

  used::Int         # capacity of live blocks
  requested::Int    # bytes requested by live blocks
  resident::Int     # size of resident segments
  queued::Int       # size of segments whose release has been queued
  failed::Int       # size of segments whose release failed
  rng::UInt64       # for the treap's priorities
end
Arena{E,L,R}() where {E,L,R} =
  Arena{E,L,R}(block_trees(Block{Fence{E,L,R}}), Dict{L,BlockTrees{Block{Fence{E,L,R}}}}(),
               Dict{R,FenceQueue{Fence{E,L,R}}}(), Dict{UInt,Block{Fence{E,L,R}}}(),
               Base.IdSet{Segment}(), Segment[], E[], 0, 0, 0, 0, 0, 0x9e3779b97f4a7c15)

function mark_wholly_free!(arena::Arena, seg::Segment)
  push!(arena.wholly_free, seg)
  seg.free_slot = length(arena.wholly_free)
  return
end
function unmark_wholly_free!(arena::Arena, seg::Segment)
  i = seg.free_slot
  i == 0 && return
  last = pop!(arena.wholly_free)
  if last !== seg
    arena.wholly_free[i] = last
    last.free_slot = i
  end
  seg.free_slot = 0
  return
end

blocktype(::Arena{E,L,R}) where {E,L,R} = Block{Fence{E,L,R}}

function next_priority!(arena::Arena)
  # xorshift64
  x = arena.rng
  x ⊻= x << 13
  x ⊻= x >> 7
  x ⊻= x << 17
  arena.rng = x
end

# what the back-end needs to provide for fences
function fence_completed end    # (event) -> whether the event has completed
function lane_retired end       # (lane) -> whether all work submitted to the lane has
                                #           finished, e.g., because the stream was recycled

# insert a free block in the right index (it isn't indexed yet)
function index!(arena::Arena, b::Block)
  class = Int(b.segment.class) + 1
  f = b.fence
  if f === nothing
    push!(arena.completed[class], b)
    b.index = COMPLETED_INDEX
  else
    trees = get!(() -> block_trees(blocktype(arena)), arena.lanes, f.lane)
    push!(trees[class], b)
    link_block!(f, b)
    b.index = LANE_INDEX
  end
  return
end
function unindex!(arena::Arena, b::Block)
  class = Int(b.segment.class) + 1
  if b.index == COMPLETED_INDEX
    delete!(arena.completed[class], b)
  elseif b.index == LANE_INDEX
    f = b.fence
    trees = arena.lanes[f.lane]
    delete!(trees[class], b)
    all(isempty, trees) && delete!(arena.lanes, f.lane)
    unlink_block!(f, b)
  end
  b.index = NOT_INDEXED
  return
end

# whether a free block of this size can serve a request of `cap` bytes
fits(size::Int, cap::Int) = size >= cap && (size < OVERSIZE || size - cap <= size >> 3)

"""
    take_block!(arena, sz, lane) -> Union{Nothing,Block}

Allocate a block for `sz` bytes to be used on `lane`, from the free blocks that can be used
there without waiting. Returns `nothing` if there are none.
"""
function take_block!(arena::Arena{E,L,R}, sz::Int, lane::L) where {E,L,R}
  cap = block_capacity(sz)
  class = Int(segment_class(cap)) + 1
  # (the smallest block that fits is also the only one an oversize block could be, as
  # larger ones fit even less)
  b = smallest_fit(arena.completed[class], cap)
  b === nothing || fits(b.size, cap) || (b = nothing)
  trees = get(arena.lanes, lane, nothing)
  if trees !== nothing
    p = smallest_fit(trees[class], cap)
    if p !== nothing && fits(p.size, cap) && (b === nothing || p.size < b.size)
      b = p
    end
  end
  b === nothing && return nothing
  unindex!(arena, b)
  split_block!(arena, b, cap)
  # reusing a block on its lane discharges its fence. one that doesn't guard any blocks
  # anymore doesn't need to be waited for.
  f = b.fence
  b.fence = nothing
  drop_unused_fence!(arena, f)
  hand_out!(arena, b, sz)
  return b
end

# split a free block (that isn't indexed) so that it has `cap` bytes. the rest keeps its
# fence, and is indexed.
function split_block!(arena::Arena, b::Block, cap::Int)
  rest = b.size - cap
  (rest < split_threshold(b.segment.class) || b.size >= OVERSIZE) && return
  r = blocktype(arena)(b.segment, b.addr + cap, rest, next_priority!(arena))
  r.fence = b.fence
  r.prev = b
  r.next = b.next
  b.next === nothing || (b.next.prev = r)
  b.next = r
  b.size = cap
  index!(arena, r)
  coalesce!(arena, r)
  return
end

function hand_out!(arena::Arena, b::Block, sz::Int)
  b.live = true
  b.requested = sz
  arena.used += b.size
  arena.requested += sz
  seg = b.segment
  seg.nlive == 0 && unmark_wholly_free!(arena, seg)
  seg.nlive += 1
  arena.live[b.addr] = b
  return
end

"""
    add_segment!(arena, base, size, sz) -> Block

Add a newly allocated segment of `size` bytes at `base`, and allocate a block for `sz`
bytes from it. If this throws, the arena is unchanged.
"""
function add_segment!(arena::Arena, base::UInt, size::Int, sz::Int; payload=nothing,
                      now::UInt64=time_ns())
  cap = block_capacity(sz)
  @assert size >= cap
  class = segment_class(cap)
  seg = Segment(base, size, class, RESIDENT, nothing, 0, now, 0, payload)
  b = blocktype(arena)(seg, base, size, next_priority!(arena))
  seg.head = b
  rest = size - cap
  r = if rest >= split_threshold(class) && size < OVERSIZE
    blocktype(arena)(seg, base + cap, rest, next_priority!(arena))
  else
    nothing
  end
  # everything that allocates comes before the arena is modified
  sizehint!(arena.live, length(arena.live) + 1)
  push!(arena.segments, seg)

  arena.resident += size
  if r !== nothing
    b.size = cap
    b.next = r
    r.prev = b
    index!(arena, r)
  end
  hand_out!(arena, b, sz)
  return b
end

"""
    new_fence!(arena, event, lane, recorder, seq) -> Fence

Register a fence, to free blocks with that are guarded by it.
"""
function new_fence!(arena::Arena{E,L,R}, event::E, lane::L, recorder::R,
                    seq::Integer) where {E,L,R}
  F = Fence{E,L,R}
  f = Fence(event, lane, recorder, seq)
  queue = get!(() -> FenceQueue{F}(nothing, nothing), arena.fences, recorder)
  # (fences are usually registered in the order they were recorded)
  after = queue.tail
  while after !== nothing && after.seq > f.seq
    after = after.prev
  end
  f.prev = after
  f.next = after === nothing ? queue.head : after.next
  f.prev === nothing ? (queue.head = f) : (f.prev.next = f)
  f.next === nothing ? (queue.tail = f) : (f.next.prev = f)
  return f
end

"""
    free_block!(arena, addr, fence; now) -> Block

Free the live block at `addr`, which is guarded by `fence` (or `nothing` if its last use
has completed).
"""
function free_block!(arena::Arena{E,L,R}, addr::UInt, fence::Union{Nothing,Fence{E,L,R}};
                     now::UInt64=time_ns()) where {E,L,R}
  b = pop!(arena.live, addr)
  b.live = false
  arena.used -= b.size
  arena.requested -= b.requested
  b.requested = 0
  # (a fence that has already been found completed doesn't need to be waited for)
  b.fence = fence === nothing || fence.done ? nothing : fence
  index!(arena, b)
  coalesce!(arena, b)
  seg = b.segment
  seg.nlive -= 1
  if seg.nlive == 0
    seg.free_since = now
    mark_wholly_free!(arena, seg)
  end
  return b
end

# the fence for two neighbouring free blocks, if they can be merged: all uses of both blocks
# need to be guarded by a single fence. a block whose last use has completed isn't merged
# with one that is still pending, which would keep it from being used on other streams.
function merged_fence(x::Block, y::Block)
  fx, fy = x.fence, y.fence
  fx === nothing && fy === nothing && return (true, nothing)
  (fx === nothing || fy === nothing) && return (false, nothing)
  covers(fx, fy) && return (true, fx)
  covers(fy, fx) && return (true, fy)
  return (false, nothing)
end

# merge an indexed free block with its indexed free neighbours where their fences allow
# it, returning the resulting block. (a free block that isn't indexed is being handed out)
function coalesce!(arena::Arena, b::Block)
  n = b.next
  if n !== nothing && n.index != NOT_INDEXED
    ok, f = merged_fence(b, n)
    if ok
      fb, fn = b.fence, n.fence
      unindex!(arena, b)
      unindex!(arena, n)
      b.size += n.size
      b.next = n.next
      n.next === nothing || (n.next.prev = b)
      n.prev = n.next = nothing
      n.size = 0
      b.fence = f
      index!(arena, b)
      fb === f || drop_unused_fence!(arena, fb)
      fn === f || drop_unused_fence!(arena, fn)
    end
  end
  p = b.prev
  if p !== nothing && p.index != NOT_INDEXED
    ok, f = merged_fence(p, b)
    if ok
      fp, fb = p.fence, b.fence
      unindex!(arena, p)
      unindex!(arena, b)
      p.size += b.size
      p.next = b.next
      b.next === nothing || (b.next.prev = p)
      b.prev = b.next = nothing
      b.size = 0
      p.fence = f
      index!(arena, p)
      fp === f || drop_unused_fence!(arena, fp)
      fb === f || drop_unused_fence!(arena, fb)
      b = p
    end
  end
  return b
end

# the fence's uses have completed: its blocks can be used everywhere
function complete_fence!(arena::Arena, f::Fence)
  f.done = true
  blocks = fence_blocks(f)
  for b in blocks
    # (an earlier one may have been merged into this one)
    b.index == LANE_INDEX && b.fence === f || continue
    unindex!(arena, b)
    b.fence = nothing
    index!(arena, b)
  end
  for b in blocks
    b.size > 0 && b.index == COMPLETED_INDEX && coalesce!(arena, b)
  end
  return
end

# a fence that doesn't guard any blocks anymore (e.g., because they were reused on its
# lane) doesn't need to be waited for
function drop_unused_fence!(arena::Arena, f::Union{Nothing,Fence})
  f === nothing && return
  f.queued && f.nblocks == 0 && f.holds == 0 && unqueue_fence!(arena, f)
  return
end

# the fence isn't waited for anymore
function unqueue_fence!(arena::Arena, f::Fence)
  queue = arena.fences[f.recorder]
  p, n = f.prev, f.next
  p === nothing ? (queue.head = n) : (p.next = n)
  n === nothing ? (queue.tail = p) : (n.prev = p)
  f.prev = f.next = nothing
  queue.head === nothing && delete!(arena.fences, f.recorder)
  f.queued = false
  f.holds == 0 && push!(arena.spare_events, f.event)
  return
end

"""
    poll_fences!(arena; exhaustive=false)

Find fences that have completed, making the blocks they guard usable everywhere. Fences
are checked in the order they were recorded per stream, until finding one that hasn't
completed, or with `exhaustive`, all of them. Returns whether any blocks became usable.
"""
function poll_fences!(arena::Arena; exhaustive::Bool=false)
  found = false
  for recorder in collect(keys(arena.fences))
    f = arena.fences[recorder].head
    while f !== nothing
      next = f.next
      if f.done || lane_retired(f.lane) || fence_completed(f.event)
        found |= f.nblocks > 0
        complete_fence!(arena, f)
      elseif !exhaustive
        break
      else
        f = next
        continue
      end
      unqueue_fence!(arena, f)
      f = next
    end
  end
  return found
end


## releasing segments

"""
    evict_segments!(arena, bytes; now, min_age=0) -> (segments, fences)

Detach wholly free segments, largest first, until at least `bytes` are being released.
Only considers segments that have been wholly free for at least `min_age` nanoseconds.
Returns the segments, and the fences that their release needs to be ordered after (once
per segment that a fence guards blocks of). Those fences are held until passing them to
`unhold_fences!`. The caller then needs to call `segment_released!` or
`segment_release_failed!` for every segment. If this throws, the arena is unchanged.
"""
function evict_segments!(arena::Arena{E,L,R}, bytes::Integer; now::UInt64=time_ns(),
                         min_age::Integer=0) where {E,L,R}
  F = Fence{E,L,R}
  segments = Segment[]
  fences = F[]
  bytes > 0 || return segments, fences
  candidates = Segment[seg for seg in arena.wholly_free if now - seg.free_since >= min_age]
  sort!(candidates; by=seg -> seg.size, rev=true)
  # select the segments and their fences before detaching anything
  for seg in candidates
    bytes > 0 || break
    push!(segments, seg)
    first = length(fences) + 1
    b = seg.head
    while b !== nothing
      b = b::blocktype(arena)
      f = b.fence
      if f !== nothing && !any(g -> g === f, view(fences, first:length(fences)))
        push!(fences, f)
      end
      b = b.next
    end
    bytes -= seg.size
  end
  for f in fences
    f.holds += 1
  end
  for seg in segments
    b = seg.head
    while b !== nothing
      b = b::blocktype(arena)
      unindex!(arena, b)
      b = b.next
    end
    unmark_wholly_free!(arena, seg)
    delete!(arena.segments, seg)
    seg.state = RELEASE_QUEUED
    arena.resident -= seg.size
    arena.queued += seg.size
  end
  return segments, fences
end

# the release of segments has been ordered after these fences
function unhold_fences!(arena::Arena, fences::Vector{<:Fence})
  for f in fences
    f.holds -= 1
    f.holds == 0 && !f.queued && push!(arena.spare_events, f.event)
  end
  return
end

function segment_released!(arena::Arena, seg::Segment)
  @assert seg.state == RELEASE_QUEUED
  seg.state = RELEASED
  arena.queued -= seg.size
  return
end

function segment_release_failed!(arena::Arena, seg::Segment)
  @assert seg.state == RELEASE_QUEUED
  seg.state = RELEASE_FAILED
  arena.queued -= seg.size
  arena.failed += seg.size
  return
end

# the bytes of free blocks, and of wholly free segments
function free_bytes(arena::Arena)
  wholly = sum(seg -> seg.size, arena.wholly_free; init=0)
  return (; free=arena.resident - arena.used, wholly_free=wholly)
end


## growth credit
#
# the hard memory limit covers all segments of a device, also in other contexts, including
# ones that are being allocated and ones whose release hasn't completed yet. credit for
# growth is taken before allocating a segment, and only returned once it has been released.

mutable struct GrowthCredit
  Base.@atomic reserved::Int
end
GrowthCredit() = GrowthCredit(0)

# take credit for `bytes`, unless that would exceed `limit` (0 for no limit)
function take_credit!(credit::GrowthCredit, bytes::Integer, limit::Integer)
  reserved = Base.@atomic credit.reserved += bytes
  if limit > 0 && reserved > limit
    Base.@atomic credit.reserved -= bytes
    return false
  end
  return true
end
return_credit!(credit::GrowthCredit, bytes::Integer) = (Base.@atomic credit.reserved -= bytes; nothing)

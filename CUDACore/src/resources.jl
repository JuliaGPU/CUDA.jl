# lifetime management of GPU resources
#
# many CUDA API calls that release resources (freeing memory without a stream-ordered pool,
# unregistering host memory, unloading modules, destroying some library objects) wait for
# all running kernels to finish, and block kernel launches from other threads meanwhile. so
# resources are never released from finalizers, which run on whatever thread happens to
# collect garbage. instead, finalizers registered with `resource_finalizer` only retire the
# resource, by pushing it onto a lock-free list (see lifecycle.jl), and regular tasks release
# it later on: when allocating, around synchronization, and periodically.
#
# what that involves depends on the resource. memory is freed in stream order where
# possible, after the work on the stream that last used it (which may have been destroyed
# in the meantime, see streams.jl). owners of memory that the GPU may still be using are
# kept alive until that work has finished, and memory that graphs use until the last graph
# that leases it has been destroyed (see `release` in managed.jl). releases that may wait
# for the GPU are held until memory is reclaimed, i.e., when running out of memory or when
# calling `reclaim()`. releases that fail keep their resource alive instead of being
# retried, as they may already have partially succeeded.
#
# routine releases never wait for the GPU, and no lock is held while waiting for the GPU
# (as the work it waits for may depend on another task that needs that lock). while a graph
# is being captured, nothing is released (only retired), as that could invalidate the capture.

include("resources/pools.jl")
include("resources/reclaim.jl")
include("resources/managed.jl")
include("resources/lifecycle.jl")
include("resources/streams.jl")
include("resources/backends.jl")
include("resources/pins.jl")

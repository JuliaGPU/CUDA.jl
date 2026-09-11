# Calling host functions from kernels

CUDA.jl kernels can call statically identifiable Julia functions on the host through the
*hostcall* mechanism. This is intended for uncommon control paths, such as error reporting. Calls cross the
CPU–GPU interconnect and share a single host service thread, so they are unsuitable for
bulk data transfer.

```julia
load(i) = DATA[i]                       # named host function

function kernel(out)
    i = threadIdx().x
    out[i] = @hostcall load(i)::Float32             # blocking call, returns a value
    @hostcall async=true println("thread ", i)      # fire-and-forget
    return
end

@cuda threads=64 kernel(out)
synchronize()       # asynchronous calls have completed, output has been printed
```

## API

Two layers are available, both built on the same protocol.

- [`@hostcall f(args...)::R`](@ref @hostcall) calls statically identifiable host functions:
  named functions, and isbits functors or closures (captured values
  are shipped along with the arguments). The return type annotation is required, `@ccall`-style;
  `@hostcall async=true f(args...)` returns after submitting its arguments and implies
  `R === Nothing`. Arguments
  may be annotated (`a::T`) to convert them before shipping. Functional forms:
  `hostcall(f, R, args...)` and `hostcall_async(f, args...)`.
- Raw ports (`hostcall_open`, `hostcall_send!`, `hostcall_recv!`, `hostcall_close!`) for
  library code that wants to stream data through the 64-byte per-lane packets itself.
  These are warp-collective. A scalar tier (`hostcall_send_scalar!`,
  `hostcall_call_scalar!`) sends a single packet from one elected lane — fire-and-forget
  or as a blocking round trip — without any warp intrinsics. The elected lane may be in
  divergent code; on pre-Volta hardware, several lanes of one converged warp must not use
  this tier independently. The runtime uses it for exception and out-of-memory reports.

Raw-port users supply a matching host handler and arrange polling and draining. Automatic
target registration and launch arming apply only to high-level calls.

## Semantics and rules

- **Warp-collective**: all lanes that reach a call site together share one port, and every
  lane submits its own arguments and receives its own result. Divergent lanes simply make
  separate calls. Calls are serviced in no particular order.
- **Values are shipped in their Julia layout** (values larger than a packet are split over
  several packets). Arguments may contain compiler-relocated host constants such as string
  literals, but arbitrary Julia references are unsupported. Pointers to device memory
  (`pointer(arr)`) arrive on the host as `CuPtr`; device arrays themselves are not converted,
  so pass their pointer and size explicitly. Results must be isbits (or `Nothing`); the
  handler's return value is converted to `R`. Non-isbits return types are rejected when the
  kernel is compiled.
- **Handlers normally run on a dedicated host thread** — a foreign thread that does not depend on
  Julia's thread pools, so hostcalls make progress even with `-t1` and while the launching
  thread is blocked in the driver. The handler runs with the kernel's context active and a
  dedicated non-blocking stream as its task-local stream. Synchronization may also run
  pending handlers on the synchronizing task, with the same context and stream. Handlers may use the CUDA API on that
  stream (e.g. copy device memory to the host), but:
  - they must not synchronize the device, or wait for work on the stream the calling kernel
    runs on: the kernel is waiting for the handler;
  - they must not compile or load kernels (loading a module synchronizes the device), so
    only call kernels that have been compiled before, and prefer `cuMemAllocAsync`-style
    stream-ordered allocations (CUDA.jl's array allocations are);
  - they must not wait on Julia tasks or conditions, or perform libuv-backed I/O
    (`println`, files, `run`) while the launching thread may be blocked in a CUDA call
    (julia#55525). The `print` family (`print`, `println`, `printstyled`, `show`, `display`)
    called directly as a hostcall target is special-cased: its output is queued and written
    at the next `synchronize()`, or earlier by a printer task when thread 1 is free.
- **Handlers run in the latest world**, like `invokelatest`: redefining a handler takes
  effect immediately — without recompiling the kernel, and including for kernels that are
  already running. Internally, calls are dispatched like Julia's `invoke`: the compiled
  kernel identifies each target by a literal pointer to its rooted key type (cached images
  carry a relocation that is re-resolved on load), and the service thread calls `jl_invoke`
  with a cached `MethodInstance` that is re-resolved when the world moves. Concrete
  dispatch signatures use the cached method; payloads such as type-valued arguments that
  require dispatch on the received value use ordinary Julia dispatch.
- **Errors**: an exception thrown by a handler, an unknown target, or a result that cannot be
  converted stops all lanes of a blocking call on the device (like a device-side exception) and is
  rethrown as a [`HostcallException`](@ref CUDACore.HostcallException) at the next stream, event,
  or device synchronization, which also completes pending asynchronous calls and flushes queued
  output. An asynchronous caller continues running; its handler failures are reported at
  synchronization.
- **Asynchronous calls** never send a result back to the device. Submission can still wait
  for a free port and, for arguments larger than a packet, for the host to receive each
  chunk. Asynchronous does not mean wait-free.
- **Nesting**: a handler cannot itself wait for another hostcall (e.g. by launching a kernel
  that hostcalls and synchronizing it), since the server is busy running the handler.
- **Graphs**: kernels replayed from a captured graph are serviced by a 1 ms heartbeat
  instead of the armed polling loop, so their calls see millisecond latency.
- **Launching**: launch kernels that use the high-level API through `@cuda` or a callable
  `HostKernel`, which arms the server while the kernel runs. Direct driver-level launches
  bypass that integration and may deadlock on a blocking call.

## Exception reporting

CUDA.jl itself uses hostcall to report device-side exceptions: the runtime library sends the
exception name, reason and (with `-g2`) stack frames through the hostcall area, without
waiting for the host, and `synchronize()` attaches the decoded report to the `KernelException`
it throws (see [Debugging](@ref DebuggingKernels)). This needs no registration, so it also works for kernels
compiled during package precompilation. Out-of-memory failures of device-side allocations
are reported the same way: the size of the failed allocation is attached to the
`OutOfMemoryError` that follows it. Every kernel that can throw therefore refers to a small
hostcall area (64 ports), which is created on first use in each context. Hostcall is core
infrastructure and cannot be disabled: exception reporting depends on it, as will other
functionality built on top of it.

## Performance

Latency depends on the CPU, GPU, interconnect, handler, and number of participating lanes.
Each lane invokes its handler separately; one warp-level request can therefore require 32
Julia calls. Use `perf/hostcall.jl` to measure launch overhead, blocking round trips and
asynchronous submission plus synchronization on the target system.

Kernels using the high-level API also enqueue a completion callback after each launch.
Ordinary kernels only carry the exception descriptor and do not arm the server. While
armed, the server polls using one CPU core, backing off to short sleeps when idle. Without
active launches it wakes every millisecond to service exception reports and graph replays.

## Multiple devices

Hostcall areas are per context, created lazily and sized for their device (the default
number of ports is the number of resident warps of that device, so heterogeneous GPUs get
differently sized areas), and a single server thread services the areas of all devices,
switching to the calling kernel's context for every call. Consequences:

- The server is a resource shared by all devices: the latency of a call grows with the
  total number of warps waiting for service across all devices.
- Exceptions are reported per context. `synchronize()`, `synchronize(stream)` and
  `device_synchronize()` throw the [`HostcallException`](@ref CUDACore.HostcallException)s
  and `KernelException`s of the context they synchronize, and the exception names the
  device; with the usual pattern of one task per device, each task sees the errors of its
  own kernels. Errors in the server thread itself are reported by whichever synchronization
  comes first.
- Synchronizing any device does complete pending asynchronous calls of all devices,
  running their handlers with their own context active.

## Configuration

The `hostcall_ports` preference (set with `Preferences.set_preferences!(CUDACore, ...)` and
restart; the preference belongs to the `CUDACore` package, not `CUDA`) sets a lower bound
on the number of ports (warp-level call slots) per context. CUDA.jl always allocates at
least one per resident warp to avoid depending on GPU forward progress; the resulting
area uses about 8 MiB of pinned memory on a large GPU. Contexts start with a small area
until a kernel that calls host functions is linked.

## Display watchdogs

On devices with a display watchdog, a kernel blocked in a hostcall counts as running. A
slow handler can therefore push the kernel over the watchdog limit, like any other
long-running kernel, so handlers should avoid long or unbounded waits.

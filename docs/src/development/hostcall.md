# Calling host functions from kernels

CUDA.jl kernels can call arbitrary Julia functions on the host through the *hostcall*
mechanism. This is meant for "unlikely" paths in device code (reporting, logging, loading
data on demand, error handling), not as a bulk data path: a call costs a few microseconds
and goes through a small packet per thread.

```julia
load(i) = DATA[i]                       # any host function

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

Three layers are available, all built on the same protocol.

- [`@hostcall f(args...)::R`](@ref @hostcall) calls any host function whose value can be
  recovered from its type: named functions, and isbits functors or closures (captured values
  are shipped along with the arguments). The return type annotation is required, `@ccall`-style;
  `@hostcall async=true f(args...)` returns immediately and implies `R === Nothing`. Arguments
  may be annotated (`a::T`) to convert them before shipping. Functional forms:
  `hostcall(f, R, args...)` and `hostcall_async(f, args...)`.
- [`HostFunction(f, R, Tuple{argtypes...})`](@ref HostFunction) registers any callable — a
  closure over host data, a function chosen at run time — and returns a handle that can be
  passed to a kernel, where it becomes a `DeviceHostFunction` that is callable like a function
  (`hf(args...)`) or usable with `@hostcall hf(args...)::R`. Targets remain registered until
  `close` is called; call it only after all kernels using the handle have completed. Dropping
  the handle without closing it leaves the target registered because an already-launched kernel
  retains only the numeric device handle.
- Raw ports (`hostcall_open`, `hostcall_send!`, `hostcall_recv!`, `hostcall_close!`) for
  library code that wants to stream data through the 64-byte per-lane packets itself.

## Semantics and rules

- **Warp-collective**: all lanes that reach a call site together share one port, and every
  lane submits its own arguments and receives its own result. Divergent lanes simply make
  separate calls. Calls are serviced in no particular order.
- **Arguments and results are isbits values**, shipped in their Julia layout (values larger
  than a packet are split over several packets). Pointers to device memory (`pointer(arr)`)
  arrive on the host as `CuPtr`; device arrays themselves are not converted, so pass their
  pointer and size explicitly. Results must be isbits (or `Nothing`); the handler's return
  value is converted to `R`.
- **Handlers run on a dedicated host thread** — a foreign thread that does not depend on
  Julia's thread pools, so hostcalls make progress even with `-t1` and while the launching
  thread is blocked in the driver. The handler runs with the kernel's context active and a
  dedicated non-blocking stream as its task-local stream. Handlers may use the CUDA API on that
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
- **Errors**: an exception thrown by a handler, an unknown target, or a result that cannot be
  converted stops all lanes of that call on the device (like a device-side exception) and is
  rethrown as a [`HostcallException`](@ref CUDACore.HostcallException) at the next stream, event,
  or device synchronization, which also completes pending asynchronous calls and flushes queued
  output. A handler exception therefore poisons every lane of the warp-level call.
- **Asynchronous calls** never send a result back to the device. For a `HostFunction` handle,
  its declared return type is used only by blocking calls.
- **Nesting**: a handler cannot itself wait for another hostcall (e.g. by launching a kernel
  that hostcalls and synchronizing it), since the server is busy running the handler.
- **Graphs**: kernels replayed from a captured graph are serviced by a 1 ms heartbeat
  instead of the armed polling loop, so their calls see millisecond latency.

## Exception reporting

CUDA.jl itself uses hostcall to report device-side exceptions: the runtime library sends the
exception name, reason and (with `-g2`) stack frames through the hostcall area, without
waiting for the host, and `synchronize()` attaches the decoded report to the `KernelException`
it throws (see [Debugging](@ref DebuggingKernels)). This needs no registration, so it also works for kernels
compiled during package precompilation. Every kernel that can throw therefore refers to a small
hostcall area (64 ports), which is created on first use in each context; when hostcall is
unavailable, the device falls back to printing the report with `printf`.

## Performance

A blocking call costs roughly 4–5 µs on a PCIe system (most of it PCIe latency: each mailbox
probe is ~0.5 µs), an asynchronous one ~3 µs; one host thread services on the order of a
million warp-level calls per second. A kernel that uses hostcalls pays a few microseconds of
extra launch overhead (the server is *armed* for its duration and disarmed by a host function
enqueued after it); kernels that do not use hostcalls are unaffected. While armed, the server
thread polls (one CPU core), backing off to short sleeps when nothing happens; when idle it
sleeps, waking up every millisecond to service stragglers.

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

Preferences (set with `Preferences.set_preferences!(CUDACore, ...)` and restart; the
preferences belong to the `CUDACore` package, not `CUDA`):
- `hostcall` (default `true`): disable the mechanism entirely; kernels using it fail to
  link, and device exceptions fall back to `printf`-based reporting.
- `hostcall_ports`: the number of ports (warp-level call slots) per context; the default is the
  number of resident warps of the device (~8 MiB of pinned memory on a large GPU). Contexts
  start with a small area until a kernel that calls host functions is linked.

## Display watchdogs

On devices with a display watchdog, a kernel blocked in a hostcall counts as running. A
slow handler can therefore push the kernel over the watchdog limit, like any other
long-running kernel. Hostcalls remain enabled by default on these devices, but handlers
should avoid long or unbounded waits.

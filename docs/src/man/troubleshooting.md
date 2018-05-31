# Troubleshooting

To increase logging verbosity of the CUDAnative compiler, launch Julia with the
`JULIA_DEBUG` environment variable set to `CUDAnative`.


## LLVM IR generated for ... is not GPU compatible

Not all of Julia is supported by CUDAnative. Several commonly-used features,
like strings or exceptions, will not compile to GPU code, because of their
interactions with the CPU-only runtime library.

When not using GPU-incompatible language features, you might still run into this
compiler error when your code contains type instabilities or other dynamic
behavior. These are often easily spotted by prefixing the failing function call
with one of several `@device_code` macros.

For example, say we define and execute the following kernel:

```julia
julia> kernel(a) = @inbounds a[threadId().x] = 0
kernel (generic function with 1 method)

julia> @cuda kernel(CuArray([1]))
ERROR: LLVM IR generated for Kernel(CuDeviceArray{Int64,1,CUDAnative.AS.Global}) is not GPU compatible
```

When running with `JULIA_DEBUG=CUDAnative`, you will get to see the actual
incompatible IR constructs. Prefixing our kernel invocation with
`@device_code_warntype` reveals our issue:

```julia
julia> @device_code_warntype @cuda kernel(CuArray([1]))
Variables:
  a::CuDeviceArray{Int64,1,CUDAnative.AS.Global}
  val<optimized out>

Body:
  begin
      Core.SSAValue(1) = (Main.threadId)()::ANY
      Core.SSAValue(2) = (Base.getproperty)(Core.SSAValue(1), :x)::ANY
      (Base.setindex!)(a::CuDeviceArray{Int64,1,CUDAnative.AS.Global}, 0, Core.SSAValue(2))::ANY
      return 0
  end::Int64
ERROR: LLVM IR generated for Kernel(CuDeviceArray{Int64,1,CUDAnative.AS.Global}) is not GPU compatible
```

Because of a typo, the call to `threadId` is untyped and returns `Any` (it
should have been `threadIdx`). In the future, we expect to be able to catch such
errors automatically.

If you want to dump all forms of generated code to disk, for further inspection,
have a look at the `@device_code` macro instead.


## Debug info and line-number information

LLVM's NVPTX back-end does not support the undocumented PTX debug format, so we cannot
generate the necessary DWARF sections. This means that debugging generated code with e.g.
`cuda-gdb` will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit
debug info (which corresponds with `nvcc -G`) when the Julia debug info level is 2 or
higher (`julia -g2`).

We do however support emitting line number information, which is useful for other CUDA tools
like `cuda-memcheck`. The functionality (which corresponds with `nvcc -lineinfo`) is enabled
when the Julia debug info level is 1 (the default value). It can be disabled by passing `-g0`
instead.

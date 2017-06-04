# Troubleshooting

You can enable verbose logging using two environment variables:

* `DEBUG`: if set, enable additional (possibly costly) run-time checks, and some more
  verbose output
* `TRACE`: if set, the `DEBUG` level will be activated, in addition with a trace of every
  call to the underlying library

In order to avoid run-time cost for checking the log level, these flags are implemented by
means of global constants. As a result, you **need to run Julia with precompilation
disabled** if you want to modify these flags:

```
$ TRACE=1 julia --compilecache=no examples/vadd.jl
TRACE: CUDAnative.jl is running in trace mode, this will generate a lot of additional output
...
```

Enabling colors with `--color=yes` is also recommended as it color-codes the output.


## `trap` and kernel launch failures

Exceptions, like the ones being thrown from out-of-bounds accesses, currently just generate
a `trap` instruction which halts the GPU. This might show up as a kernel launch failure, or
an unrelated error in another API call.

If the error is thrown from an array access, and an out-of-bounds access is suspected, it is
useful to turn of bounds checking (`julia --check-bounds=no`) and run the Julia process
under `cuda-memcheck` while enabling debug mode 1 (the default value) or higher. This way,
`cuda-memcheck` will be able to accurately pinpoint the out-of-bounds access, while
specifying the exact location of the access within the active grid and block.


## `code_*` alternatives

CUDAnative provides alternatives to Base's `code_llvm` and `code_native` to inspect
generated GPU code:

```julia
julia> foo(a, i) = (a[1] = i; return nothing)
foo (generic function with 1 method)

julia> a = CuArray{Int}(1)

julia> CUDAnative.@code_llvm foo(a, 1)

; Function Attrs: nounwind
define i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {
...
}

julia> @code_ptx foo(a, 1)
.visible .entry julia_foo_62419(
        .param .u64 julia_foo_62419_param_0,
        .param .u64 julia_foo_62419_param_1
)
{
...
}

julia> @code_sass foo(a, 1)
        code for sm_20
                Function : julia_foo_62539
...
```

Non-macro versions of these reflection entry-points are available as well (ie. `code_llvm`,
etc), but as there's type conversions happening behind the scenes you will need to take care
and perform those conversions manually:

```julia
julia> CUDAnative.code_llvm(foo, (CuArray{Int,1},Int))
ERROR: error compiling foo: ...

julia> CUDAnative.code_llvm(foo, (CuDeviceArray{Int,1},Int))

; Function Attrs: nounwind
define i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {
...
}
```


## Debug info and line-number information

LLVM's NVPTX back-end does not support the undocumented PTX debug format, so we cannot
generate the necessary DWARF sections. This means that debugging generated code with e.g.
`cuda-gdb` will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit
debug info (which corresponds with `nvcc -g`) when the Julia debug info level is 2 or
higher (`julia -g2`).

We do however support emitting line number information, which is useful for other CUDA tools
like `cuda-memcheck`. The functionality (which corresponds with `nvcc -lineinfo`) is enabled
when the Julia debug info level is 1 (the default value) or higher.

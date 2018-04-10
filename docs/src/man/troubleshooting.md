# Troubleshooting

To increase logging verbosity of the CUDAnative compiler, launch Julia with the
`JULIA_DEBUG` environment variable set to `CUDAnative`.


## `code_*` alternatives

CUDAnative provides alternatives to Base's `code_llvm` and `code_native` to
inspect generated device code:

```julia
julia> foo(a, i) = (a[1] = i; return nothing)
foo (generic function with 1 method)

julia> a = CuArray{Int}(1)

julia> @device_code_llvm @cuda foo(a, 1)

; Function Attrs: nounwind
define void @ptxcall_foo_1({ [1 x i64], { i64 } }, i64) local_unnamed_addr {
...
}

julia> @device_code_ptx @cuda foo(a, 1)
.visible .entry ptxcall_foo_3(
        .param .align 8 .b8 ptxcall_foo_3_param_0[16],
        .param .u64 ptxcall_foo_3_param_1
)
{
...
}

julia> @device_code_sass foo(a, 1)
        code for sm_20
                Function : ptxcall_foo_5
...
```

Non-macro versions of these reflection entry-points are available as well (ie. `code_llvm`,
etc), but as there's type conversions happening behind the scenes you will need to take care
and perform those conversions manually:

```julia
julia> CUDAnative.code_llvm(foo, (CuArray{Int,1},Int))
ERROR: error compiling foo: ...

julia> CUDAnative.code_llvm(foo, (CuDeviceArray{Int,1,AS.Global},Int))

; Function Attrs: nounwind
define void @julia_foo_35907({ [1 x i64], { i64 } }, i64) local_unnamed_addr {
...
}
```


## Debug info and line-number information

LLVM's NVPTX back-end does not support the undocumented PTX debug format, so we cannot
generate the necessary DWARF sections. This means that debugging generated code with e.g.
`cuda-gdb` will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit
debug info (which corresponds with `nvcc -G`) when the Julia debug info level is 2 or
higher (`julia -g2`).

We do however support emitting line number information, which is useful for other CUDA tools
like `cuda-memcheck`. The functionality (which corresponds with `nvcc -lineinfo`) is enabled
when the Julia debug info level is 1 (the default value) or higher. It can be disabled by
passing `-g0` instead.

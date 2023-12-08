# Troubleshooting

This section deals with common errors you might run into while writing GPU code, preventing
the code to compile.


## InvalidIRError: compiling ... resulted in invalid LLVM IR

Not all of Julia is supported by CUDA.jl. Several commonly-used features, like strings or
exceptions, will not compile to GPU code, because of their interactions with the CPU-only
runtime library.

For example, say we define and try to execute the following kernel:

```julia
julia> function kernel(a)
         @inbounds a[threadId().x] = 0
         return
       end

julia> @cuda kernel(CuArray([1]))
ERROR: InvalidIRError: compiling kernel kernel(CuDeviceArray{Int64,1,1}) resulted in invalid LLVM IR
Reason: unsupported dynamic function invocation (call to setindex!)
Stacktrace:
 [1] kernel at REPL[2]:2
Reason: unsupported dynamic function invocation (call to getproperty)
Stacktrace:
 [1] kernel at REPL[2]:2
Reason: unsupported use of an undefined name (use of 'threadId')
Stacktrace:
 [1] kernel at REPL[2]:2
```

CUDA.jl does its best to decode the unsupported IR and figure out where it came from. In
this case, there's two so-called dynamic invocations, which happen when a function call
cannot be statically resolved (often because the compiler could not fully infer the call,
e.g., due to inaccurate or instable type information). These are a red herring, and the real
cause is listed last: a typo in the use of the `threadIdx` function! If we fix this, the IR
error disappears and our kernel successfully compiles and executes.


## KernelError: kernel returns a value of type `Union{}`

Where the previous section clearly pointed to the source of invalid IR, in other cases your
function will return an error. This is encoded by the Julia compiler as a return value of
type `Union{}`:

```julia
julia> function kernel(a)
         @inbounds a[threadIdx().x] = CUDA.sin(a[threadIdx().x])
         return
       end

julia> @cuda kernel(CuArray([1]))
ERROR: GPU compilation of kernel kernel(CuDeviceArray{Int64,1,1}) failed
KernelError: kernel returns a value of type `Union{}`
```

Now we don't know where this error came from, and we will have to take a look ourselves at
the generated code. This is easily done using the `@device_code` introspection macros, which
mimic their Base counterparts (e.g. `@device_code_llvm` instead of `@code_llvm`, etc).

To debug an error returned by a kernel, we should use `@device_code_warntype` to inspect the
Julia IR. Furthermore, this macro has an `interactive` mode, which further facilitates
inspecting this IR using Cthulhu.jl. First, install and import this package, and then try to
execute the kernel again prefixed by `@device_code_warntype interactive=true`:

```julia
julia> using Cthulhu

julia> @device_code_warntype interactive=true @cuda kernel(CuArray([1]))
Variables
  #self#::Core.Compiler.Const(kernel, false)
  a::CuDeviceArray{Int64,1,1}
  val::Union{}

Body::Union{}
1 ─ %1  = CUDA.sin::Core.Compiler.Const(CUDA.sin, false)
│   ...
│   %14 = (...)::Int64
└──       goto #2
2 ─       (%1)(%14)
└──       $(Expr(:unreachable))

Select a call to descend into or ↩ to ascend.
 • %17  = call CUDA.sin(::Int64)::Union{}
```

Both from the IR and the list of calls Cthulhu offers to inspect further, we can see that
the call to `CUDA.sin(::Int64)` results in an error: in the IR it is immediately followed by
an `unreachable`, while in the list of calls it is inferred to return `Union{}`. Now we know
where to look, it's easy to figure out what's wrong:

```julia
help?> CUDA.sin
  # 2 methods for generic function "sin":
  [1] sin(x::Float32) in CUDA at /home/tim/Julia/pkg/CUDA/src/device/intrinsics/math.jl:13
  [2] sin(x::Float64) in CUDA at /home/tim/Julia/pkg/CUDA/src/device/intrinsics/math.jl:12
```

There's no method of `CUDA.sin` that accepts an Int64, and thus the function was determined
to unconditionally throw a method error. For now, we disallow these situations and refuse to
compile, but in the spirit of dynamic languages we might change this behavior to just throw
an error at run time.

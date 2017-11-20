# Hacking

## Generated functions

Generated functions are used heavily in CUDAnative.jl, in combination with LLVM.jl, to
generate type-specialized code and IR. If evaluating the generator results in an error,
Julia generates a dynamic call to the generator for you to inspect the error at run-time.
This is a problem in the world of GPUs, where dynamic calls are prohibited. A band-aid is to
print the exception during inference:

```patch
diff --git a/base/inference.jl b/base/inference.jl
index 6443665676..b03d78ddaa 100644
--- a/base/inference.jl
+++ b/base/inference.jl
@@ -2430,7 +2430,10 @@ function typeinf_frame(linfo::MethodInstance, caller, optimize::Bool, cached::Bo
             try
                 # user code might throw errors – ignore them
                 src = get_staged(linfo)
-            catch
+            catch ex
+                println("WARNING: An error occurred during generated function execution.")
+                println(ex)
+                ccall(:jlbacktrace, Void, ())
                 return nothing
             end
         else
```


## Adding intrinsics

Adding intrinsics to `CUDAnative.jl` can be relatively convoluted, depending on the type of
intrinsic. Most of the boil down to inlining a snippet of LLVM IR, using `llvmcall` (or
`ccall` with the `llvmcall` calling convention). For more complex code, use LLVM.jl to build
the IR string.


### `libdevice` intrinsics

These intrinsics are represented by function calls to `libdevice`. Most of them should
already be covered. There's a convenience macro, `@wrap`, simplifying the job of adding and
exporting intrinsics, and converting arguments and return values. See the documentation of
the macro for more details, and look at `src/device/libdevice.jl` for examples.


### LLVM back-end intrinsics

Calls to functions like `llvm.nvvm.barrier0` are backed the PTX LLVM back-end, and can be
wrapped using `ccall` with the `llvmcall` calling convention. For more complex intrinsics,
or when you're not actually calling an intrinsic function, you can still use `@wrap`.


### Inline PTX assembly

When there's no corresponding `libdevice` function or PTX back-end intrinsic exposing the
required functionality, you can use inline PTX assembly via `llvmcall`. This requires you to
embed the PTX assembly in LLVM IR, which is often messy.

If the source of the assembly instructions is CUDA C code, you simplify this task by first
compiling the CUDA code using Clang, and adapting the resulting LLVM IR for use within
`llvmcall`. For example, extracting the following function definition from the CUDA SDK:

```cuda
__device__ unsigned int __ballot(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.ballot.b32 \t%0, %%p1; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}
```

We can generate the following LLVM IR by executing `clang++ -Xclang -fcuda-is-device -S
-emit-llvm -target nvptx64 ballot.cu -o -` (you might need to add [some CUDA
boilerplate](https://gist.github.com/eliben/b014ac17cbe5a452803f)):

```
define i32 @_Z8__balloti(i32 %a) #0 {
  %1 = alloca i32, align 4
  %result = alloca i32, align 4
  store i32 %a, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  %3 = call i32 asm sideeffect "{ \0A\09.reg .pred \09%p1; \0A\09setp.ne.u32 \09%p1, $1, 0; \0A\09vote.ballot.b32 \09$0, %p1; \0A\09}", "=r,r"(i32 %2) #1, !srcloc !1
  store i32 %3, i32* %result, align 4
  %4 = load i32, i32* %result, align 4
  ret i32 %4
}
```

Finally, cleaning this code up we end up with the following `llvmcall` invocation:

```julia
ballot_asm = """{
   .reg .pred %p1;
   setp.ne.u32 %p1, \$1, 0;
   vote.ballot.b32 \$0, %p1;
}"""

function ballot(pred::Bool)
    return Base.llvmcall(
        """%2 = call i32 asm sideeffect "$ballot_asm", "=r,r"(i32 %0)
           ret i32 %2""",
        UInt32, Tuple{Int32}, convert(Int32, pred))
end
```

In the future, we will use LLVM.jl to properly embed inline assembly instead of this
string-based hackery.


### Other functionality

For other functionality, like shared memory, or when some additional management is required,
like storing a global variable for `printf`'s formatting string, you should use LLVM.jl to
build the IR code instead of hacking strings together. As this doesn't touch global state,
you can even do so from a `@generated` function. Do take care however to use Julia's LLVM
context for all operations.

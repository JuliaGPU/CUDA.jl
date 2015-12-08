Address sanitizer
-----------------

Running CUDA code under Address Sanitizer is tricky in general, as [CUDA
allocates fixed memory overlapping with ASAN's internal data
structures](https://github.com/google/sanitizers/issues/629). The symptom of
this issue is `cuInit(0)` returning `CUDA_ERROR_OUT_OF_MEMORY` (code 2), or ASAN
complaining about memory accesses to its shadow gap. Julia further complicates
this issue as it uses LLVM on multiple levels (initial compilation, and run-time
JIT).


## Build patched LLVM toolchain

Start out by building a patched version of LLVM and Clang, including
compiler-rt. Make sure that you select a version which is compatible with Julia,
both to compile with and link to. Apply the following patch, which refrains from
starting the ASAN shadow gap at `0x7FFF8000` and reverts to the old
(prelink-incompatible) behaviour:

```patch
--- lib/Transforms/Instrumentation/AddressSanitizer.cpp (revision 255005)
+++ lib/Transforms/Instrumentation/AddressSanitizer.cpp (working copy)
@@ -359,7 +359,7 @@
       if (IsKasan)
         Mapping.Offset = kLinuxKasan_ShadowOffset64;
       else
-        Mapping.Offset = kSmallX86_64ShadowOffset;
+        Mapping.Offset = kDefaultShadowOffset64;
     } else if (IsMIPS64)
       Mapping.Offset = kMIPS64_ShadowOffset64;
     else if (IsAArch64)
--- projects/compiler-rt/lib/asan/asan_mapping.h (revision 255005)
+++ projects/compiler-rt/lib/asan/asan_mapping.h (working copy)
@@ -146,7 +146,7 @@
 #  elif SANITIZER_IOS
 #    define SHADOW_OFFSET kIosShadowOffset64
 #  else
-#   define SHADOW_OFFSET kDefaultShort64bitShadowOffset
+#   define SHADOW_OFFSET kDefaultShadowOffset64
 #  endif
 # endif
 #endif
```

Build the toolchain, preferably enabling debug information and assertions, but
make sure you use the CMake build system rather than autotools ([compiler-rt
only works with CMake](https://llvm.org/bugs/show_bug.cgi?id=22757)). Do not
generate shared libraries, as [CMake does not generate the required
`libllvm-x.y.so` Julia will be looking
for](https://llvm.org/bugs/show_bug.cgi?id=15493). Also don't forget to enable
the `NVPTX` target.


## Build Julia

Next, we'll build and link a version of Julia using this patched toolchain. It
is advisable to create an out-of-tree build directory for this type of build, as
it will be too slow for general use.

Configure Julia to use the patched LLVM version both for compilation and to link
against. For example, you could put the following options in your `Make.user`:

```
USECLANG=1
override CC=$(WHEREVER)/llvm-x.y.debug+asserts/bin/clang
override CXX=$(WHEREVER)/llvm-x.y.debug+asserts/bin/clang++

override USE_SYSTEM_LLVM=1
override USE_LLVM_SHLIB=0
override LLVM_CONFIG=$(WHEREVER)/llvm-x.y.debug+asserts/bin/llvm-config
```

Due to [a problem with Julia's
`Makefile`s](https://github.com/JuliaLang/julia/issues/13858), you first need to
build the Julia dependencies without enabling ASAN. Afterwards, you can rebuild
Julia with `SANITIZE=1`, optionally putting that value in `Make.user` for ease
of use.

Note that compilation will probably fail due to leaked memory. You can
circumvent this by exporting `LSAN_OPTIONS=exitcode=0` at the shell. This can be
embedded in `Make.user` using `export LSAN_OPTIONS=exitcode=0`, but also needs
to be defined at run-time (ie. when running regular Julia code after building).

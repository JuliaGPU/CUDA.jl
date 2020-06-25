Wrapping headers
================

This directory contains scripts that can be used to automatically generate
wrappers for C headers by NVIDIA, such as CUBLAS or CUDNN. This is done using
Clang.jl, with some CSTParser.jl-based scripts to clean-up the result.

In CuArrays.jl, the wrappers need to know whether pointers passed into the
library point to CPU or GPU memory (i.e. `Ptr` or `CuPtr`). This information is
not available from the headers, and will be prompted for.



Usage
-----

Either run `wrap.jl` directly, or include it using Revise.jl and call the
`main()` function. Be sure to activate the project environment in this folder.

The script will try to discover the global CUDA include path, and certain
preconfigured headers within. If you need to override these paths, you can use
the `CUDA_PATH` environment variable to steer CUDAapi.jl into discovering a
different CUDA installation, or use per-library variables to override the
include path (e.g. `CUTENSOR_INCLUDE`).


### Entering pointer type information

When running the script, any new functions with unknown pointer types (i.e.,
ones that are missing from the `pointers.json` database) will prompt for
additional information. For example, removing `cusparseSpruneDense2csrNnz` from
the database we get to see the following prompt when running `main()`:

```
cusparseSpruneDense2csrNnz(cusparseHandle_t         handle,
                           int                      m,
                           int                      n,
                           const float*             A,
                           int                      lda,
                           const float*             threshold,
                           const cusparseMatDescr_t descrC,
                           int*                     csrRowPtrC,
                           int*                     nnzTotalDevHostPtr,
                           void*                    pBuffer);

cusparseSpruneDense2csrNnz
- argument 4: A::Ptr{Cfloat}
- argument 6: threshold::Ptr{Cfloat}
- argument 8: csrRowPtrC::Ptr{Cint}
- argument 9: nnzTotalDevHostPtr::Ptr{Cint}
- argument 10: pBuffer::Ptr{Cvoid}
```

This prompt first tries to grab the definition from the relevant header, and
then prints the function name again, together with the arguments that need
additional type information (i.e. pointer-valued arguments). To figure out what
type of pointers we're dealing with, the snippet from the header might give some
information, you probably would need to look at the NVIDIA documentation, or if
that is lacking you should Google for example code. Here, it is obvious from the
snippet that `nnzTotalDevHostPtr` is a dual GPU/CPU pointer. Looking at the
documentation, there's a table that indicates `percentage` is a CPU pointer, and
the others point to GPU memory. We enter this as follows:

```
GPU pointers> 4 8 10
Dual GPU/CPU pointers> 9
```

The remaining argument, 6, is typed as a CPU pointer. This results in the
following entry in the database:

```json
"cusparseSpruneDense2csrNnz": {
    "handle": null,
    "m": null,
    "n": null,
    "A": "CuPtr",
    "lda": null,
    "threshold": "Ptr",
    "descrC": null,
    "csrRowPtrC": "CuPtr",
    "nnzTotalDevHostPtr": "PtrOrCuPtr",
    "pBuffer": "CuPtr"
}
```

And the following wrapper:

```julia
function cusparseSpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC, csrRowPtrC,
                                    nnzTotalDevHostPtr, pBuffer)
    @check ccall((:cusparseSpruneDense2csrNnz, libcusparse), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr,
                 pBuffer)
end
```

Most functions aren't as complex, and to facilitate these cases there are
shortcuts you can use at the `GPU pointers` prompt:

- Entering `0` matches _all_ arguments
- Entering negative numbers matches all arguments _except_ the negative ones

These shortcuts are only valid ad the `GPU pointers` prompt, and cannot be
combined with regular entries (i.e. positive, non-null numbers).


### Manual patches

After going through this for all libraries, the wrappers are written to the
`CuArrays/src/LIBRARY` folder, with an additional copy in `CuArrays/res/wrap`.
You should now make sure the wrappers work as expected. This might not be the
case, as Clang.jl can have difficulties wrapping certain C constructs. Edit the
wrappers in `CuArrays/src` until they work, but keep the edits as limited as
possible.

Now `diff -u` those headers to the raw versions in `res/wrap`, and write the
patch to the `patches` directory within. Prefix the filename with the name of
the library. Next time you generate headers, these patches will be automatically
applied, ensuring no more manual edits are required anymore.

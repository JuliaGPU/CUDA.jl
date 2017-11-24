# Usage

Quick start:

```jldoctest
using CUDAdrv

dev = CuDevice(0)
ctx = CuContext(dev)

# code that does GPU computations

destroy!(ctx)

# output

```

```@meta
DocTestSetup = quote
    using CUDAdrv

    dev = CuDevice(0)
    ctx = CuContext(dev)
end
```


## Automatic memory management

Except for the encapsulating context, `destroy` or `unload` calls are never needed. Objects
are registered with the Julia garbage collector, and are automatically finalized when they
go out of scope.

However, many CUDA API functions implicitly depend on global state, such as the current
active context. The wrapper needs to model those dependencies in order for objects not to
get destroyed before any dependent object is. If we fail to model these dependency
relations, API calls might randomly fail, eg. in the case of a missing context dependency
with a `INVALID_CONTEXT` or `CONTEXT_IS_DESTROYED` error message.

If this seems to be the case, re-run with `TRACE=1` and file a bug report.

## Arrays

### Device arrays

Device arrays are called `CuArray`s, as opposed to regular (CPU) Julia `Array`s

`CuArray`s can be initialized with regular `Array`s:

```jldoctest
A   = zeros(Float32,3,4)
d_A = CuArray(A)

# output

3×4 Array{Float32,2}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
```

The `d_` syntax is a conventional way of reminding yourself that the array is
allocated on the device.

```@meta
DocTestSetup = quote
    using CUDAdrv

    dev = CuDevice(0)
    ctx = CuContext(dev)

    A   = zeros(Float32,3,4)
    d_A = CuArray(A)
end
```

To copy a device array back to the host, use:
```jldoctest
copy!(A, d_A)

# output

3×4 Array{Float32,2}:
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0
```

You can also invert `d_A` and `A` to copy from host to device.

Most of the typical Julia functions, like `size`, `ndims`, `eltype`, etc.,  work
on CuArrays. One noteworthy omission is that you can't directly index a
CuArray: `d_A[2,4]` will fail. This is not supported because host/device memory transfers
are relatively slow, and you don't want to write code that (on the host side) makes use of
individual elements in a device array. If you want to inspect the values in a device array,
first use `copy!` to copy it to host memory.

## Modules and custom kernels

This will not teach you about CUDA programming; for that, please refer to the CUDA
documentation and other online sources.

### Compiling your own modules

You can write and use your own custom kernels, first writing a `.cu` file and compiling it
as a `ptx` module. On Linux, compilation would look something like this:

```
nvcc -ptx mycudamodule.cu
```

You can specify that the code should be compiled for compute capability 2.0 devices or
higher using:

```
nvcc -ptx -gencode=arch=compute_20,code=sm_20 mycudamodule.cu
```

If you want to write code that will support multiple datatypes (e.g., `Float32` and
`Float64`), it's recommended that you use C++ and write your code using templates. Then use
`extern C` to instantiate bindings for each datatype. For example:

```cpp
template <typename T>
__device__ void kernel_function1(T *data) {
    // Code goes here
}
template <typename T1, typename T2>
__device__ void kernel_function2(T1 *data1, T2 *data2) {
    // Code goes here
}

extern "C"
{
    void __global__ kernel_function1_float(float *data) {kernel_function1(data);}
    void __global__ kernel_function1_double(double *data) {kernel_function1(data);}
    void __global__ kernel_function2_int_float(int *data1, float *data2) {kernel_function2(data1,data2);}
}
```

#### Initializing and freeing PTX modules

To easily make your kernels available, the recommended approach is to define something
analogous to the following for each `ptx` module (this example uses the kernels described in
the previous section):

```julia
module MyCudaModule

import CUDAdrv
import CUDAdrv: CuModule, CuModuleFile, CuFunction, cudacall

export function1

const ptxdict = Dict()
const mdlist = Array{CuModule}(0)

function mdinit(devlist)
    global ptxdict
    global mdlist
    isempty(mdlist) || error("mdlist is not empty")
    for dev in devlist
        CuDevice(dev)
        md = CuModuleFile("mycudamodule.ptx")

        ptxdict[("function1", Float32)] = CuFunction(md, "kernel_function1_float")
        ptxdict[("function1", Float64)] = CuFunction(md, "kernel_function1_double")
        ptxdict[("function2", Int32, Float32)] = CuFunction(md, "kernel_function2_int_float")

        push!(mdlist, md)
    end
end

mdclose() = (empty!(mdlist); empty!(ptxdict))

function finit()
  mdclose()
end

function init(devlist)
  mdinit(devlist)
end

function function1(griddim::CuDim, blockdim::CuDim, data::CuArray{T}) where T
    cufunction1 = ptxdict[("function1", T)]
    cudacall(cufunction1, griddim, blockdim, (Ptr{T},), data)
end

...

end  # MyCudaModule
```

Usage will look something like the following:

```julia
gpuid = 0
dev = CuDevice(gpuid) # Or the ID of the GPU you want, if you have many of them
ctx = CuContext(dev)

MyCudaModule.init(gpuid)
# Code that uses functions from your MyCudaModule
MyCudaModule.finit()

destroy!(ctx)
```

# Other notes

## Notes on memory

Julia convention is that matrices are stored in column-major order, whereas C (and CUDA) use
row-major. For efficiency this wrapper avoids reordering memory, so that the linear sequence
of addresses is the same between main memory and the GPU. For most usages, this is probably
what you want.

However, for the purposes of linear algebra, this effectively means that one is storing the
transpose of matrices on the GPU. Keep this in mind when manipulating code on your GPU
kernels.



## Troubleshooting

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
TRACE: CUDAdrv.jl is running in trace mode, this will generate a lot of additional output
...
```

Enabling colors with `--color=yes` is also recommended as it color-codes the output.

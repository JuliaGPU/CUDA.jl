# CUDAdrv.jl

*A Julia wrapper for the CUDA driver API.*

**Build status**: [![][buildbot-julia05-img]][buildbot-julia05-url] [![][buildbot-julia06-img]][buildbot-julia06-url] [![][buildbot-juliadev-img]][buildbot-juliadev-url]

**Documentation**: [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url]

**Code coverage**: [![][coverage-img]][coverage-url]

[buildbot-julia05-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAdrv-julia05-x86-64bit&name=julia%200.5
[buildbot-julia05-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAdrv-julia05-x86-64bit
[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAdrv-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAdrv-julia06-x86-64bit
[buildbot-juliadev-img]: http://ci.maleadt.net/shields/build.php?builder=CUDAdrv-juliadev-x86-64bit&name=julia%20dev
[buildbot-juliadev-url]: http://ci.maleadt.net/shields/url.php?builder=CUDAdrv-juliadev-x86-64bit

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: http://juliagpu.github.io/CUDAdrv.jl/stable
[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: http://juliagpu.github.io/CUDAdrv.jl/latest

[coverage-img]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl/coverage.svg
[coverage-url]: https://codecov.io/gh/JuliaGPU/CUDAdrv.jl

This package aims to provide high-level wrappers for the functionality exposed by the CUDA
driver API, and is meant for users who need high- or low-level access to the CUDA toolkit or
the underlying hardware.

## Installation

CUDAdrv is a registered package, and can be installed using the Julia package manager:

```julia
Pkg.add("CUDAdrv")
```

Julia versions 0.5 and 0.6 are supported, with limited effort to keep the package working on
current master as well. Refer to [the documentation][docs-stable-url] for more information
on how to install or use this package.

## Usage

Start by saying `using CUDAdrv`, or `import CUDAdrv` if you prefer to qualify everything with
the module name.

### GPU initialization

One or more GPUs can be initialized, used for computations, and freed for other uses.
Memory management is automatic, in the sense that GPU objects are registered
with the Julia garbage collector.

To use a GPU, use the syntax:

```julia
dev = CuDevice(0) # Or the ID of the GPU you want, if you have many of them
ctx = CuContext(dev)
# Code that does GPU computation
destroy!(ctx)
```

### Arrays

#### Device arrays

Device arrays are called `CuArray`s, as opposed to regular (CPU) Julia `Array`s

`CuArray`s can be initialized with regular `Array`s:

```julia
A   = zeros(Float32,3,4)
d_A = CuArray(A)
```
The `d_` syntax is a conventional way of reminding yourself that the array is
allocated on the device.

To copy a device array back to the host, use either of
```julia
copy!(A, d_A)
```

Most of the typical Julia functions, like `size`, `ndims`, `eltype`, etc.,  work
on CuArrays. One noteworthy omission is that you can't directly index a
CuArray: `d_A[2,4]` will fail. This is not supported because host/device memory transfers
are relatively slow, and you don't want to write code that (on the host side) makes use of
individual elements in a device array. If you want to inspect the values in a device array,
first use `copy!` to copy it to host memory.

### Modules and custom kernels

This will not teach you about CUDA programming; for that, please refer to the CUDA
documentation and other online sources.

#### Compiling your own modules

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

function finit( )
  mdclose()
end

function init( devlist )
  mdinit( devlist )
end

function function1{T}(griddim::CuDim, blockdim::CuDim, data::CuArray{T})
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

MyCudaModule.init( gpuid )
# Code that uses functions from your MyCudaModule
MyCudaModule.finit()

destroy!(ctx)
```


## Random notes

### Notes on memory

Julia convention is that matrices are stored in column-major order, whereas C (and CUDA) use
row-major. For efficiency this wrapper avoids reordering memory, so that the linear sequence
of addresses is the same between main memory and the GPU. For most usages, this is probably
what you want.

However, for the purposes of linear algebra, this effectively means that one is storing the
transpose of matrices on the GPU. Keep this in mind when manipulating code on your GPU
kernels.

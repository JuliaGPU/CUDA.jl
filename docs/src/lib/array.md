# Arrays

CUDAdrv provides a primitive but useful array type to manage GPU data organized in an
plain, dense fashion.

```@docs
CUDAdrv.CuArray
CUDAdrv.copy!{T}(::CuArray{T}, ::Array{T})
CUDAdrv.copy!{T}(::Array{T}, ::CuArray{T})
CUDAdrv.copy!{T}(::CuArray{T}, ::CuArray{T})
CUDAdrv.CuArray{T,N}(::Array{T,N})
CUDAdrv.Array{T,N}(::CuArray{T,N})
```

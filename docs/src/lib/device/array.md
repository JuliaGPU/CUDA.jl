# Arrays

CUDAnative provides a primitive, lightweight array type to manage GPU data organized in an
plain, dense fashion. This is the device-counterpart to CUDAdrv's `CuArray`, and implements
(part of) the array interface as well as other functionality for use _on_ the GPU:

```@docs
CUDAnative.CuDeviceArray
CUDAnative.ldg
```

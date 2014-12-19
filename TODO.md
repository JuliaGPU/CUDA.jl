* Document environment variables (CUDA_FORCE_API_VERSION, CUDA_FORCE_GPU_TRIPLE,
  CUDA_FORCE_GPU_ARCH)

* Use a DevicePtr containing a Ptr (cfr CppPtr in
  [Cxx.jl](https://github.com/Keno/Cxx.jl/blob/master/src/Cxx.jl))

* Pass the native codegen context into `@cuda`, allowing for multiple active
  contexts

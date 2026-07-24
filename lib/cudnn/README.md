# cuDNN.jl

Julia wrapper for [NVIDIA cuDNN](https://developer.nvidia.com/cudnn), providing
GPU-accelerated neural-network primitives for `CuArray` values.

cuDNN.jl is part of the [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) repository. The
wrapper is organized around cuDNN's backend and graph APIs:

- `libcudnn.jl` contains generated bindings for the raw C API.
- `backend.jl` wraps `cudnnBackendDescriptor_t` with typed descriptor helpers.
- `graph/` provides a Julia frontend with `Graph`, `Tensor`, operation factories,
  heuristics, plan caching, and execution.
- `ops/` exposes the Julia-facing operations used by downstream packages, such as
  `attention!`, `convolution!`, pooling, and batch normalization.

Fixed-function compatibility wrappers live in `src/legacy`, with matching tests in
`test/legacy`. Softmax, dropout, and RNN remain outside `legacy`.

See the CUDA.jl manual's [cuDNN wrapper design](https://cuda.juliagpu.org/stable/lib/cudnn/)
section for the detailed layering, migration notes, and debugging hints.

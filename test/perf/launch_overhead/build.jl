using CUDAapi
using CUDAdrv

dev = CuDevice(0)
cap = capability(dev)

cd(@__DIR__) do
    toolkit = CUDAapi.find_toolkit()
    nvcc = CUDAapi.find_cuda_binary("nvcc", toolkit)
    toolchain = CUDAapi.find_toolchain(toolkit)
    flags = `-ccbin=$(toolchain.host_compiler) -arch=sm_$(cap.major)$(cap.minor)`
    run(`$nvcc $flags -ptx -o cuda.ptx cuda.cu`)
    run(`$nvcc $flags -lm -lcuda -o cuda cuda.c`)
end

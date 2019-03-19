# wrappers for the libcudadevrt library
#
# The libcudadevrt library is a collection of PTX bitcode functions that implement parts of
# the CUDA API for execution on the device, such as device synchronization primitives,
# dynamic kernel APIs, etc.

export device_synchronize

"""
    device_synchronize()

Wait for the device to finish. This is the device side version,
and should not be called from the host. 

`device_synchronize` acts as a synchronization point for
child grids in the context of dynamic parallelism.
"""
@inline device_synchronize() = @wrap cudaDeviceSynchronize()::i32

using CUDA, Base.Test

# TODO: add all bugs or issues here, with appropriate error/assertion checks

dev = CuDevice(0)
ctx = CuContext(dev)

initialize_codegen(ctx, dev)

@target ptx kernel_empty() = return nothing
@cuda (1, 1) kernel_empty()

@test_throws @cuda (1, 1) SomeModule.SomeKernel()

using CUDA, Base.Test

# TODO: add all bugs or issues here, with appropriate error/assertion checks

dev = CuDevice(0)
ctx = CuContext(dev)

initialize_codegen(ctx, dev)

@target ptx kernel_empty() = return nothing
@test_throws ErrorException @eval begin
    @cuda (0, 0) kernel_empty()
end
@eval begin
    @cuda (1, 1) kernel_empty()
end

@test_throws @cuda (1, 1) SomeModule.SomeKernel()

@compile reference_dummy """
__global__ void reference_dummy()
{
}
"""

@target ptx function kernel_dummy()
    return nothing
end

@compile dummy """
__global__ void dummy()
{
}
"""

@target ptx function kernel_dummy()
    return nothing
end

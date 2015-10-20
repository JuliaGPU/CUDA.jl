@target ptx function kernel_scalaradd(a::CuDeviceArray{Float32}, x)
    i = blockIdx_x() + (threadIdx_x()-1) * gridDim_x()
    a[i] = a[i] + x

    return nothing
end

# TODO: get and compare dim tuple instead of xyz
@target ptx function kernel_lastvalue(a::CuDeviceArray{Float32},
                                      x::CuDeviceArray{Float32})
    i = blockIdx_x() + (threadIdx_x()-1) * gridDim_x()
    max = gridDim_x() * blockDim_x()
    if i == max
        x[1] = a[i]
    end

    return nothing
end

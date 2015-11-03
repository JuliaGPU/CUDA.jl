# TODO: get and compare dim tuple instead of xyz

@target ptx function kernel_lastvalue(a::CuDeviceArray{Float32},
                                      x::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    max = gridDim().x * blockDim().x
    if i == max
        x[1] = a[i]
    end

    return nothing
end


@target ptx function kernel_lastvalue_devfun(a::CuDeviceArray{Float32},
                                             x::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    max = gridDim().x * blockDim().x
    if i == max
        x[1] = lastvalue_devfun(a, i)
    end

    return nothing
end

@target ptx function lastvalue_devfun(a::CuDeviceArray{Float32}, i)
    return a[i]
end

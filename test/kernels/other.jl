@target ptx function kernel_scalaradd(a::CuDeviceArray{Float32}, x)
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    a[i] = a[i] + x

    return nothing
end

# TODO: get and compare dim tuple instead of xyz
@target ptx function kernel_lastvalue(a::CuDeviceArray{Float32},
                                      x::CuDeviceArray{Float32})
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    max = numBlocks_x() * numThreads_x()
    if i == max
        x[1] = a[i]
    end

    return nothing
end

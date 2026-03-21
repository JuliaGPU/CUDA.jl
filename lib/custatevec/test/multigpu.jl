@testset "cuStateVec multiGPU" begin

    nGlobalBits  = 2;
    nLocalBits   = 2;
    nSubSvs      = 2^nGlobalBits
    subSvSize    = 2^nLocalBits
    bitStringLen = 2
    bitOrdering  = [1, 0]

    bitString = Vector{Int}(undef, bitStringLen)
    bitString_result = zeros(Int, bitStringLen)
    # the most random of all numbers
    randnum = 0.71

    h_sv = Vector{ComplexF64}[]
    push!(h_sv, [0.0; 0.125im; 0.250im; 0.375im])
    push!(h_sv, [0.0; -0.125im; -0.250im; -0.375im])
    push!(h_sv, [0.125; 0.125-0.125im; 0.125-0.250im; 0.125-0.375im])
    push!(h_sv, [-0.125; -0.125-0.125im; -0.125-0.250im; -0.125-0.375im])

    h_sv_result = Vector{ComplexF64}[]
    push!(h_sv_result, zeros(ComplexF64, subSvSize))
    push!(h_sv_result, zeros(ComplexF64, subSvSize))
    push!(h_sv_result, ComplexF64[1/√2; 0; 0; 0])
    push!(h_sv_result, ComplexF64[-1/√2; 0; 0; 0])

    n_devices = 4;
    # on CI, if we only have a single device, set up multiple devices
    # so that we properly cover the multigpu code paths.
    if ndevices() < n_devices
        sv_devices = fill(device(), n_devices)
    else
        sv_devices = collect(devices())[1:n_devices]
    end
    initial_dev = device()
    d_sv = similar(h_sv, CuStateVec{ComplexF64})
    normArray = similar(d_sv, Float64)
    try
        for sv_i in 1:length(d_sv)
            device!(sv_devices[sv_i])
            d_sv[sv_i] = CuStateVec(h_sv[sv_i])
            normArray[sv_i] = abs2SumArray(d_sv[sv_i], Int[], Int[], Int[])[]
        end
    finally
        device!(initial_dev)
    end
    cumulativeArray = zeros(Float64, length(normArray) + 1)
    for sv_i in 1:length(normArray)
        cumulativeArray[sv_i+1] = cumulativeArray[sv_i] + normArray[sv_i]
    end
    try
        for sv_i in 1:length(d_sv)
            if cumulativeArray[sv_i] <= randnum && randnum < cumulativeArray[sv_i + 1]
                norm = cumulativeArray[end]
                offset = cumulativeArray[sv_i]
                device!(sv_devices[sv_i])
                new_sv, bitstring = batchMeasureWithOffset!(d_sv[sv_i], bitOrdering, randnum, offset, norm)
                @test length(bitstring) == nLocalBits
            end
        end
    finally
        device!(initial_dev)
    end
end

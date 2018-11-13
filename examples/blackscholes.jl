# EXCLUDE FROM TESTING

using CuArrays, CUDAnative, CUDAdrv, CUDAapi

CUDAnative.initialize()
const dev = device()
const cap = capability(dev)

using BenchmarkTools
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10
BenchmarkTools.DEFAULT_PARAMETERS.gcsample = true

using SpecialFunctions


## scalar CPU version

@inline cndf2(in::Float32) = 0.5f0 + 0.5f0 * erf(0.707106781f0 * in)

function blackscholes_cpu(sptprice::Float32, strike::Float32, rate::Float32,
                      volatility::Float32, time::Float32)
    logterm = log10(sptprice / strike)
    powterm = .5f0 * volatility * volatility
    den = volatility * sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike * exp(-rate * time)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    return call - futureValue + sptprice
end


## vectorized CPU version

@inline cndf2(in::AbstractArray{Float32}) = 0.5f0 .+ 0.5f0 .* erf.(0.707106781f0 .* in)

function blackscholes_cpu(sptprice::AbstractArray{Float32},
                      strike::AbstractArray{Float32},
                      rate::AbstractArray{Float32},
                      volatility::AbstractArray{Float32},
                      time::AbstractArray{Float32})
    logterm = log10.(sptprice ./ strike)
    powterm = .5f0 .* volatility .* volatility
    den = volatility .* sqrt.(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2(d1)
    NofXd2 = cndf2(d2)
    futureValue = strike .* exp.(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    return call .- futureValue .+ sptprice
end


## native CUDA version

@inline cndf2_cuda(in::Float32) = 0.5f0 + 0.5f0 * CUDAnative.erf(0.707106781f0 * in)

function blackscholes_kernel(sptprice::AbstractArray{Float32},
                             strike::AbstractArray{Float32},
                             rate::AbstractArray{Float32},
                             volatility::AbstractArray{Float32},
                             time::AbstractArray{Float32},
                             out::AbstractArray{Float32})
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x

    if i <= size(sptprice, 1)
        logterm = CUDAnative.log10(sptprice[i] / strike[i])
        powterm = 0.5f0 * volatility[i] * volatility[i]
        den = volatility[i] * CUDAnative.sqrt(time[i])
        d1 = (((rate[i] + powterm) * time[i]) + logterm) / den
        d2 = d1 - den
        NofXd1 = cndf2_cuda(d1)
        NofXd2 = cndf2_cuda(d2)
        futureValue = strike[i] * CUDAnative.exp(-rate[i] * time[i])
        c1 = futureValue * NofXd2
        call = sptprice[i] * NofXd1 - c1
        out[i] = call - futureValue + sptprice[i]
    end

    return
end


## scalar CuArrays version

function blackscholes_gpu(sptprice::Float32, strike::Float32, rate::Float32,
                          volatility::Float32, time::Float32)
    logterm = CUDAnative.log10(sptprice / strike)
    powterm = .5f0 * volatility * volatility
    den = volatility * CUDAnative.sqrt(time)
    d1 = (((rate + powterm) * time) + logterm) / den
    d2 = d1 - den
    NofXd1 = cndf2_cuda(d1)
    NofXd2 = cndf2_cuda(d2)
    futureValue = strike * CUDAnative.exp(-rate * time)
    c1 = futureValue * NofXd2
    call = sptprice * NofXd1 - c1
    return call - futureValue + sptprice
end


## vectorized CuArrays version

@inline cndf2_cuarr(in::AbstractArray{Float32}) = 0.5f0 .+ 0.5f0 .* CUDAnative.erf.(0.707106781f0 .* in)

function blackscholes_gpu(sptprice::AbstractArray{Float32},
                          strike::AbstractArray{Float32},
                          rate::AbstractArray{Float32},
                          volatility::AbstractArray{Float32},
                          time::AbstractArray{Float32})
    logterm = CUDAnative.log10.(sptprice ./ strike)
    powterm = .5f0 .* volatility .* volatility
    den = volatility .* CUDAnative.sqrt.(time)
    d1 = (((rate .+ powterm) .* time) .+ logterm) ./ den
    d2 = d1 .- den
    NofXd1 = cndf2_cuarr(d1)
    NofXd2 = cndf2_cuarr(d2)
    futureValue = strike .* CUDAnative.exp.(- rate .* time)
    c1 = futureValue .* NofXd2
    call = sptprice .* NofXd1 .- c1
    return call .- futureValue .+ sptprice
end


## non-native CUDA C version

const cuda_source = "$(tempname()).cu"
const cuda_ptx = "$(tempname()).ptx"

open(cuda_source, "w") do io
    print(io, """
        extern "C" __global__ void blackscholes_kernel(const float *sptprice,
                                                       const float *strike,
                                                       const float *rate,
                                                       const float *volatility,
                                                       const float *time,
                                                       float *out,
                                                       size_t n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float logterm = log10(sptprice[i] / strike[i]);
                float powterm = 0.5 * volatility[i] * volatility[i];
                float den = volatility[i] * sqrt(time[i]);
                float d1 = (((rate[i] + powterm) * time[i]) + logterm) / den;
                float d2 = d1 - den;
                float NofXd1 = 0.5 + 0.5 * erf(0.707106781 * d1);
                float NofXd2 = 0.5 + 0.5 * erf(0.707106781 * d2);
                float futureValue = strike[i] * exp(-rate[i] * time[i]);
                float c1 = futureValue * NofXd2;
                float call = sptprice[i] * NofXd1 - c1;
                out[i] = call - futureValue + sptprice[i];
            }
        }
    """)
end

toolkit = CUDAapi.find_toolkit()
nvcc = CUDAapi.find_cuda_binary("nvcc", toolkit)
toolchain = CUDAapi.find_toolchain(toolkit)
flags = `-ccbin=$(toolchain.host_compiler) -arch=sm_$(cap.major)$(cap.minor)`
run(`$nvcc $flags -ptx -o $cuda_ptx $cuda_source`)

const cuda_module = CuModuleFile(cuda_ptx)
const cuda_function = CuFunction(cuda_module, "blackscholes_kernel")


## main

function checksum(reference, result)
    reference_sum = sum(reference)
    result_sum = sum(result)
    diff = abs(1-reference_sum/result_sum)
    if diff>0.01
        warn("checksum failed: $result_sum instead of $reference_sum (relative difference: $diff)")
        println(stacktrace())
    end
end

function main(iterations)
    sptprice   = Float32[ 42.0 for i = 1:iterations ]
    strike     = Float32[ 40.0 + (i / iterations) for i = 1:iterations ]
    rate       = Float32[ 0.5 for i = 1:iterations ]
    volatility = Float32[ 0.2 for i = 1:iterations ]
    time       = Float32[ 0.5 for i = 1:iterations ]

    timings = Dict()

    reference = blackscholes_cpu.(sptprice, strike, rate, volatility, time)

    let benchmark = @benchmarkable begin
                out = blackscholes_cpu.($sptprice, $strike, $rate,
                                        $volatility, $time)
            end setup=(
                out = nothing
            ) teardown=(
                checksum($reference, out)
            )
        timings["Single-threaded (scalar)"] = run(benchmark)
    end

    let benchmark = @benchmarkable begin
                out = blackscholes_cpu($sptprice, $strike, $rate,
                                       $volatility, $time)
            end setup=(
                out = nothing
            ) teardown=(
                checksum($reference, out)
            )
        timings["Single-threaded (vectorized)"] = run(benchmark)
    end

    let benchmark = @benchmarkable begin
                cudacall(cuda_function,
                         Tuple{Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
                               Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Csize_t},
                         sptprice_dev, strike_dev, rate_dev, volatility_dev,
                         time_dev, out, n; blocks=grid, threads=block)
                synchronize()
            end setup=(
                sptprice_dev = CUDAdrv.CuArray($sptprice);
                strike_dev = CUDAdrv.CuArray($strike);
                rate_dev = CUDAdrv.CuArray($rate);
                volatility_dev = CUDAdrv.CuArray($volatility);
                time_dev = CUDAdrv.CuArray($time);
                out = CUDAdrv.CuArray{Float32}(size($strike));

                n = size($sptprice, 1);
                block = min(n, 1024);
                grid = ceil(Integer, n/block)
            ) teardown=(
                checksum($reference, Array(out))
            )
        timings["CUDA C (kernel)"] = run(benchmark)
    end

    let benchmark = @benchmarkable begin
                @cuda blocks=grid threads=block blackscholes_kernel(sptprice_dev, strike_dev, rate_dev,
                                                                    volatility_dev, time_dev, out)
                synchronize()
            end setup=(
                sptprice_dev = CUDAdrv.CuArray($sptprice);
                strike_dev = CUDAdrv.CuArray($strike);
                rate_dev = CUDAdrv.CuArray($rate);
                volatility_dev = CUDAdrv.CuArray($volatility);
                time_dev = CUDAdrv.CuArray($time);
                out = CUDAdrv.CuArray{Float32}(size($strike));

                n = size($sptprice, 1);
                block = min(n, 1024);
                grid = ceil(Integer, n/block)
            ) teardown=(
                checksum($reference, Array(out))
            )
        timings["CUDAnative.jl (kernel)"] = run(benchmark)
    end

    let benchmark = @benchmarkable begin
                out = blackscholes_gpu.(sptprice_dev, strike_dev, rate_dev,
                                        volatility_dev, time_dev)
                synchronize()
            end setup=(
                sptprice_dev = CuArrays.CuArray($sptprice);
                strike_dev = CuArrays.CuArray($strike);
                rate_dev = CuArrays.CuArray($rate);
                volatility_dev = CuArrays.CuArray($volatility);
                time_dev = CuArrays.CuArray($time);
                out = nothing
            ) teardown=(
                checksum($reference, Array(out))
            )
        timings["CuArrays.jl (scalar)"] = run(benchmark)
    end

    let benchmark = @benchmarkable begin
                out = blackscholes_gpu(sptprice_dev, strike_dev, rate_dev,
                                       volatility_dev, time_dev)
                synchronize()
            end setup=(
                sptprice_dev = CuArrays.CuArray($sptprice);
                strike_dev = CuArrays.CuArray($strike);
                rate_dev = CuArrays.CuArray($rate);
                volatility_dev = CuArrays.CuArray($volatility);
                time_dev = CuArrays.CuArray($time);
                out = nothing
            ) teardown=(
                checksum($reference, Array(out))
            )
        timings["CuArrays.jl (vectorized)"] = run(benchmark)
    end

    return timings
end

function driver()
    iterations = 10^7
    timings = main(iterations)

    println()
    println("Timings:")
    for (test, trials) in timings
        println("* $test: ", BenchmarkTools.prettytime(time(trials)))
    end

    println()
    println("Rates:")
    for (test, trials) in timings
        println("* $test: ", 1e9*iterations/time(trials), " ops/sec")
    end
end

driver()

rm(cuda_source)
rm(cuda_ptx)

using DataStructures
using CUDA
using Test

function init_case(T, f, N::Integer)
    a = map(x -> T(f(x)), 1:N)
    c = CuArray(a)
    a, c
end

function check_equivalence(a::Vector, c::Vector)
    counter(a) == counter(c) && issorted(c)
end

function check_sort!(T, N, f=identity)
    original_arr, device_arr = init_case(T, f, N)
    sort!(device_arr)
    host_result = Array(device_arr)
    check_equivalence(original_arr, host_result)
end

check() = check_sort!(UInt8, 100000, x -> round(255 * rand() ^ 2))

function main()
    for dbg = [false, true]
        println("Debug info: $dbg")
        empty!(CUDA.__device_properties)
        empty!(CUDA.cufunction_cache[device()])
        ENV["JULIA_CUDA_DEBUG_INFO"] = dbg

        path = "/tmp/dbg_$dbg"
        isdir(path) && rm(path, recursive=true)
        @device_code dir=path check()
        open("$path/kernel.sass", "w") do io
            @device_code_sass io=io check()
        end
        println("Device code written to $path")

        for x in 1:10
            print("Iteration $x\r")
            @assert check()
        end
        println()
    end
end

isinteractive() || main()

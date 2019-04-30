using CUDAapi

using Test


## library types (Accessibility and `@enum` behaviour)

@test CUDAapi.PATCH_LEVEL == CUDAapi.libraryPropertyType(2)
@test CUDAapi.C_32U == CUDAapi.cudaDataType(13)


## properties

@test !CUDAapi.gcc_supported(v"5.0", v"5.5")
@test CUDAapi.gcc_supported(v"5.0", v"8.0")
CUDAapi.devices_for_cuda(v"8.0")
CUDAapi.devices_for_llvm(v"5.0")
CUDAapi.isas_for_cuda(v"8.0")
CUDAapi.isas_for_llvm(v"5.0")


## discovery

# generic
find_binary([Sys.iswindows() ? "CHKDSK" : "true"])
find_library([Sys.iswindows() ? "NTDLL" : "c"])

# CUDA
macro test_something(ex)
    quote
        rv = $(ex)
        @test rv !== nothing
        rv
    end
end

dirs = find_toolkit()
@test !isempty(dirs)

if !isempty(dirs)
    ver = find_toolkit_version(dirs)

    @test_something find_cuda_binary("nvcc", dirs)
    @test_something find_cuda_library("cudart", dirs)
    @test_something find_cuda_library("nvtx", dirs)
    @test_something find_libdevice([v"3.0"], dirs)
    @test_something find_libcudadevrt(dirs)
    @test_something find_host_compiler()
    @test_something find_host_compiler(ver)
    @test_something find_toolchain(dirs)
    @test_something find_toolchain(dirs, ver)

    if haskey(ENV, "CI")
        # CI deals with plenty of CUDA versions, which makes discovery tricky.
        # dump a relevant tree of files to help debugging
        function traverse(dir, level=0)
            for entry in readdir(dir)
                print("  "^level)
                path = joinpath(dir, entry)
                if isdir(path)
                    println("└ $entry")
                    traverse(path, level+1)
                else
                    println("├ $entry")
                end
            end
        end
        for dir in dirs
            println("File tree of toolkit directory $dir:")
            traverse(dir)
        end
    end
end
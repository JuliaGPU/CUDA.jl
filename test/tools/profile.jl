import NVTX
using CUDA.Profile: push_row!, filtermask

@testset "profiler" begin

############################################################################################

@testset "helpers" begin

@testset "push_row!" begin
    nt = (id = Int[], name = String[], value = Union{Missing,Int}[])
    push_row!(nt, (id = 1, name = "a"))
    push_row!(nt, (id = 2, name = "b", value = 10))

    @test nt.id == [1, 2]
    @test nt.name == ["a", "b"]
    @test ismissing(nt.value[1])
    @test nt.value[2] == 10
end

@testset "filtermask" begin
    nt = (id = [1, 2, 3, 4, 5], val = [10, 25, 15, 30, 5])
    filtered = filtermask(nt, nt.val .> 15)

    @test filtered.id == [2, 4]
    @test filtered.val == [25, 30]
end

end

############################################################################################

@testset "external" begin

CUDA.Profile.start()
CUDA.Profile.stop()

@test CUDA.@profile external=true begin
    true
end

end

############################################################################################

@static if can_use_cupti()
@testset "integrated" begin

# smoke test
let
    str = string(CUDA.@profile true)
    @test occursin("No host-side activity was recorded", str)
    @test occursin("No device-side activity was recorded", str)
end

# kernel launch
let
    @cuda identity(nothing)
    str = string(CUDA.@profile @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test !occursin("cuCtxSynchronize", str)
    @test !occursin("ID", str)
end

# kernel launch (trace)
let
    str = string(CUDA.@profile trace=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test !occursin("cuCtxSynchronize", str)
end

# kernel launch (raw trace)
let
    str = string(CUDA.@profile trace=true raw=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test occursin("cuCtxSynchronize", str)
end

# benchmarked profile
let
    str = string(CUDA.@bprofile @cuda identity(nothing))
    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)
    @test !occursin("cuCtxGetCurrent", str)

    str = string(CUDA.@bprofile raw=true @cuda identity(nothing))
    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)
    @test occursin("cuCtxGetCurrent", str)
end

if CUPTI.version() != v"13.0.0" # NVIDIA/NVTX#125

# NVTX markers
let
    str = string(CUDA.@profile trace=true NVTX.@mark "a marker")
    @test occursin("NVTX marker", str)
    @test occursin("a marker", str)
end

# NVTX ranges
let
    str = string(CUDA.@profile trace=true NVTX.@range "a range" identity(nothing))
    @test occursin("NVTX ranges", str)
    @test occursin("a range", str)
end

end

end
end

end

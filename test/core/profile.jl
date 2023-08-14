@testset "profiler" begin

@testset "external" begin

CUDA.Profile.start()
CUDA.Profile.stop()

@test CUDA.@profile external=true begin
    true
end

end

############################################################################################

if VERSION >= v"1.9" && CUDA.runtime_version() >= v"11.2" && can_use_cupti()
@testset "integrated" begin

# smoke test
let
    str = sprint() do io
        rv = CUDA.@profile io=io begin
            true
        end
        @test rv
    end

    @test occursin("No host-side activity was recorded", str)
    @test occursin("No device-side activity was recorded", str)
end

# kernel launch
let
    @cuda identity(nothing)
    str = sprint() do io
        CUDA.@profile io=io begin
            @cuda identity(nothing)
        end
    end

    @test occursin("cuLaunchKernel", str)
    @test occursin("_Z8identityv", str)

    @test !occursin("cuCtxSynchronize", str)
    @test !occursin("ID", str)
end

# kernel launch (trace)
let
    str = sprint() do io
        CUDA.@profile io=io trace=true begin
            @cuda identity(nothing)
        end
    end

    @test occursin("cuLaunchKernel", str)
    @test occursin("_Z8identityv", str)

    @test occursin("ID", str)

    @test !occursin("cuCtxSynchronize", str)
end

# kernel launch (raw trace)
let
    str = sprint() do io
        CUDA.@profile io=io trace=true raw=true begin
            @cuda identity(nothing)
        end
    end

    @test occursin("cuLaunchKernel", str)
    @test occursin("_Z8identityv", str)

    @test occursin("ID", str)

    @test occursin("cuCtxSynchronize", str)
end

end
end

end

include("test_helpers.jl")


## list of tests

runner = if Main.mode == "single"
    ShuffledTestRunner()
elseif Main.mode == "parallel"
    ParallelTestRunner()
elseif Main.mode == "distributed"
    DistributedTestRunner()
else
    error("Invalid mode '$(Main.mode)")
end

@testset runner=runner "CUDA" begin
    function find_tests(dir=@__DIR__)
        for entry in readdir(dir)
            path = joinpath(dir, entry)
            if isdir(path)
                @testset "$entry" begin
                    find_tests(path)
                end
            elseif !in(entry, ["runtests.jl", "tests_early.jl", "tests_all.jl", "test_helpers.jl"]) &&
                    endswith(entry, ".jl")
                # run tests in a clean environment
                CUDA.device_reset!()

                include(path)

                # NOTE: we can't do the device reset after testing, because test outputs
                #       (e.g. exception objects, stack traces, etc) that might contain
                #       references to objects are yet to be sent to the master.
            end
        end
    end
    find_tests()

    @testset "GPUArrays" begin
        TestSuite.test(CuArray)
    end
end

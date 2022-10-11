using CUTENSOR, CUDA, Test

@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

@test CUTENSOR.has_cutensor()
@info "CUTENSOR version: $(CUTENSOR.version()) (built for CUDA $(CUTENSOR.cuda_version()))"

@testset "CUTENSOR" begin

# include all tests
for entry in readdir(@__DIR__)
    endswith(entry, ".jl") || continue
    entry in ["runtests.jl"] && continue

    # generate a testset
    name = splitext(entry)[1]
    @eval begin
        @testset $name begin
            include($entry)
        end
    end
end

end

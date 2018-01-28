@testset "documentation" begin

cd(joinpath(dirname(@__DIR__), "docs")) do
    withenv("TEST"=>true) do
        run(julia_cmd(`make.jl`))
    end
end

end

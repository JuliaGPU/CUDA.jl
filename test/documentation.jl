if "Documenter" in Pkg.available()
    @testset "documentation" begin

    out = Pipe()
    result = cd(joinpath(dirname(@__DIR__), "docs")) do
        withenv("TEST"=>true) do
            cmd = julia_cmd(`make.jl`)
            success(pipeline(cmd; stdout=out, stderr=out))
        end
    end
    close(out.in)

    output = readstring(out)
    println(output)

    if !result
        error("error making documentation")    
    end

    if contains(output, "Test Error")
        error("error running doctests")    
    end

    end
else
    warn("Documenter.jl not installed, skipping documentation tests.")
end

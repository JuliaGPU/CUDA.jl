@testset "documentation" begin

out = Pipe()
result = cd(joinpath(dirname(@__DIR__), "docs")) do
    withenv("TEST"=>true) do
        success(pipeline(`$(Base.julia_cmd()) --color=$(Base.have_color?"yes":"no") make.jl`; stdout=out, stderr=out))
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

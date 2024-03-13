using Aqua

const MAX_AMBIGUITIES = 15

# FIXME: we have plenty of ambiguities, let's at least ensure that we don't create more
let ambs = Aqua.detect_ambiguities(CUDA; recursive=true)
    pkg_match(pkgname, pkgdir::Nothing) = false
    pkg_match(pkgname, pkgdir::AbstractString) = occursin(pkgname, pkgdir)

    # StaticArrays pirates a bunch of Random stuff...
    filter!(x -> !pkg_match("StaticArrays", pkgdir(x[1].module)) &&
                 !pkg_match("StaticArrays", pkgdir(x[2].module)), ambs)

    # if we'll fail this test, at least show which ambiguities were detected
    if length(ambs) > MAX_AMBIGUITIES
        for (ma, mb) in ambs
            println()
            println("Ambiguity detected:")
            if VERSION >= v"1.9"
                print("  ")
                Base.show_method(stdout, ma)
                println()
                print("  ")
                Base.show_method(stdout, mb)
                println()
            else
                println("  ", ma)
                println("  ", mb)
            end
        end
    end
    @test length(ambs) ≤ MAX_AMBIGUITIES
end

Aqua.test_all(CUDA;
    stale_deps=(ignore=[:CUDA_Runtime_Discovery, :CUDA_Runtime_jll,
                        :SpecialFunctions],),

    # tested above
    ambiguities=false,

    # FIXME: Adapt.WrappedArray contains subtypes that do not bind the N typevar
    #Aqua.test_unbound_args(CUDA)
    unbound_args=false
)

module Latency

using CUDA
using BenchmarkTools

function main()
    results = BenchmarkGroup()

    CUDA.versioninfo()

    base_cmd = Base.julia_cmd()
    if Base.JLOptions().project != C_NULL
        base_cmd = `$base_cmd --project=$(unsafe_string(Base.JLOptions().project))`
    end
    run(`$base_cmd -e "using CUDA; device()"`)

    results
end

end

Latency.main()

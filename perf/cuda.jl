group = addgroup!(SUITE, "cuda")

let group = addgroup!(group, "synchronization")
    let group = addgroup!(group, "stream")
        group["blocking"] = @benchmarkable synchronize(blocking=true)
        group["auto"] = @benchmarkable synchronize()
        group["nonblocking"] = @benchmarkable synchronize(spin=false)
    end
    let group = addgroup!(group, "context")
        group["blocking"] = @benchmarkable device_synchronize(blocking=true)
        group["auto"] = @benchmarkable device_synchronize()
        group["nonblocking"] = @benchmarkable device_synchronize(spin=false)
    end
end

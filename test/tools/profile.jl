import JSON
import NVTX
using CUDA.Profile: push_row!, filtermask

# A100s let only one process profile them when MIG is enabled, which other CI jobs on the
# same GPU may be doing, so retry sessions that find the GPU busy
function retry_busy(f; timeout=600)
    start = time()
    while true
        try
            return f()
        catch err
            busy = occursin("being profiled by another process", sprint(showerror, err))
            busy && time() - start < timeout || rethrow()
            sleep(5)
        end
    end
end
macro retry_busy(ex)
    :(retry_busy(() -> $(esc(ex))))
end

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

@testset "detect_cupti" begin
    @test @inferred(CUDA.Profile.detect_cupti()) isa Bool
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

if CUPTI.version() < v"11"
    @test_throws "Integrated profiling requires CUPTI 11" CUDA.@profile error("must not execute")
end

@static if can_use_cupti() && CUPTI.version() >= v"11"
@testset "integrated" begin

@testset "activity cleanup" begin
    @test_throws ErrorException CUDA.@profile error("profiled code failed")
    @test CUPTI.activity_config[] === nothing
    @test CUDA.@profile(identity(nothing)) isa CUDA.Profile.ProfileResults
end

@testset "profiled allocation" begin
    a = nothing
    CUDA.@profile begin
        a = CuArray([1, 2, 3])
    end
    @test Array(a) == [1, 2, 3]
end

if CUDACore.is_tegra() && CUDA.runtime_version() < v"12"
    @testset "unsupported Tegra activity" begin
        cfg = CUPTI.ActivityConfig([CUPTI.CUPTI_ACTIVITY_KIND_DRIVER,
                                    CUPTI.CUPTI_ACTIVITY_KIND_MARKER_DATA])
        ran = Ref(false)
        CUPTI.@enable! cfg begin
            ran[] = true
            CUDA.synchronize()
        end
        @test ran[]
        @test CUPTI.activity_config[] === nothing
    end
end

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

    # without cropping, rows aren't padded to the width of the table
    @test !any(line -> endswith(line, r"[ │┐┤┘]"), split(str, '\n'))
end

# kernel launch (trace)
let
    str = string(CUDA.@profile trace=true @cuda identity(nothing))

    @test occursin("cuLaunchKernel", str)
    @test occursin("identity()", str)

    @test occursin("ID", str)

    @test !occursin("cuCtxSynchronize", str)
end

# timestamps keep their nanosecond resolution, and aren't rounded to multiples of
# 2^-22 s (~238 ns) as happens when converting CUPTI's raw timestamps to seconds
let
    results = CUDA.@profile for i in 1:10
        @cuda identity(nothing)
    end
    times = [results.host.stop .- results.host.start; results.device.stop .- results.device.start]
    @test any(t -> !isinteger(t * 2^22), times)
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

# unnamed NVTX ranges
let
    results = CUDA.@profile NVTX.range_end(NVTX.range_start())
    @test occursin("NVTX ranges", string(results))
    json = JSON.parse(repr("application/json", results))
    @test only(json["nvtx"])["name"] === nothing
end

# JSON output
let
    @cuda identity(nothing)
    results = CUDA.@profile @cuda identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["view"] == "summary"
    @test json["duration_ns"] > 0
    kernel = only(filter(row -> occursin("identity", row["name"]), json["device"]))
    @test kernel["calls"] == 1
    @test kernel["total_ns"] == kernel["mean_ns"] > 0
    @test kernel["std_ns"] === nothing
    @test any(row -> startswith(row["name"], "cuLaunchKernel"), json["host"])
    @test isempty(json["nvtx"])

    # the trace view matches API calls with the kernels they launched
    results = CUDA.@profile trace=true @cuda threads=2 blocks=3 identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["view"] == "trace"
    kernel = only(filter(row -> occursin("identity", row["name"]), json["device"]))
    @test kernel["block"] == [2, 1, 1]
    @test kernel["grid"] == [3, 1, 1]
    @test kernel["size_bytes"] === nothing
    launch = only(filter(row -> row["id"] == kernel["id"], json["host"]))
    @test startswith(launch["name"], "cuLaunchKernel")
    @test launch["start_ns"] < kernel["start_ns"]

    # names are escaped
    name = "a \"range\"\n\twith \\ characters → \e"
    results = CUDA.@profile trace=true NVTX.@range name identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["schema_version"] == 1
    @test only(json["nvtx"])["name"] == name

    # ranges that don't end have no duration
    results = CUDA.@profile trace=true NVTX.range_start(; message="open")
    json = JSON.parse(repr("application/json", results))
    @test only(json["nvtx"])["duration_ns"] === nothing

    # the raw trace includes everything
    results = CUDA.@profile trace=true raw=true @cuda identity(nothing)
    json = JSON.parse(repr("application/json", results))
    @test json["raw"]
    @test any(row -> row["name"] == "cuCtxSynchronize", json["host"])
end

end

end
end

############################################################################################

# the range profiler of CUPTI 12.8 to 13.1 doesn't work
broken_range_profiler = can_use_cupti() && v"12.8" <= CUPTI.version() < v"13.2"

# whether this device and driver let us read performance counters
can_collect_metrics = can_use_cupti() && CUPTI.has_range_profiler() &&
                      !broken_range_profiler && try
    @retry_busy CUDA.@metrics metrics=["gpu__time_duration.sum"] nothing
    true
catch err
    msg = sprint(showerror, err)
    contains(msg, "not supported on this device") || contains(msg, "performance counters") ||
        contains(msg, "did not collect hardware metrics") ||
        contains(msg, "did not record when kernels executed") || rethrow()
    @warn "Cannot collect hardware metrics, skipping tests" exception=err
    false
end

@testset "metrics" begin

@test_throws ArgumentError @macroexpand CUDA.@metrics :memory nothing

if can_use_cupti() && !CUPTI.has_range_profiler()
    @test_throws "CUDA 12.6 Update 2" CUDA.@metrics error("must not execute")
elseif broken_range_profiler
    @test_throws "CUDA 13.2" CUDA.@metrics error("must not execute")
end

if can_collect_metrics

function increment(a)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    i <= length(a) && (@inbounds a[i] += 1f0)
    return
end

a = CUDA.zeros(Float32, 1024)

@testset "overview" begin
    p = @retry_busy CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p isa CUDA.Profile.KernelProfile
    @test length(p.kernels.id) == 1
    @test p.passes >= 1
    @test size(p.values) == (1, length(p.metrics))
    @test p.values[1, 1] > 0    # duration

    str = string(p)
    @test occursin("increment", str)
    @test occursin("Compute", str)
    @test !any(line -> endswith(line, r"[ │┐┤┘]"), split(str, '\n'))
    @test !occursin("bold", str)

    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "summary"
    @test json["launches"] == 1
    @test json["passes"] == p.passes
    @test [metric["name"] for metric in json["metrics"]] == p.metrics
    @test json["metrics"][1]["unit"] == "nanosecond"
    kernel = only(json["kernels"])
    @test occursin("increment", kernel["name"])
    @test kernel["calls"] == 1
    @test kernel["time_fraction"] == 1
    @test kernel["metrics"][p.metrics[1]] == p.values[1, 1]

    p = @retry_busy CUDA.@metrics trace=true @cuda threads=32 blocks=32 increment(a)
    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "trace"
    kernel = only(json["kernels"])
    @test kernel["id"] == only(p.kernels.id)
    @test kernel["grid"] == [32, 1, 1]
    @test kernel["block"] == [32, 1, 1]

    str = string(@retry_busy CUDA.@metrics trace=true @cuda threads=32 blocks=32 increment(a))
    @test occursin("ID", str)
    @test occursin("Blocks", str)

    str = string(@retry_busy CUDA.@metrics metrics=:memory @cuda threads=32 blocks=32 increment(a))
    @test occursin("DRAM traffic", str)

    @test occursin("No kernels", string(@retry_busy CUDA.@metrics nothing))
    json = JSON.parse(repr("application/json", @retry_busy CUDA.@metrics nothing))
    @test json["launches"] == 0
    @test isempty(json["kernels"])
end

@testset "raw metrics" begin
    p = @retry_busy CUDA.@metrics metrics=["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"] begin
        @cuda threads=32 blocks=32 increment(a)
    end
    @test p.metrics == ["smsp__warps_launched.sum", "smsp__average_warp_latency.ratio"]
    @test p.values[1, 1] == 32
    # ratios aren't percentages
    @test !occursin("%", string(p))

    # without a duration, there's no time fraction, and units aren't known
    json = JSON.parse(repr("application/json", p))
    kernel = only(json["kernels"])
    @test kernel["time_fraction"] === nothing
    @test kernel["metrics"]["smsp__warps_launched.sum"] == 32
    @test all(metric -> metric["unit"] === nothing, json["metrics"])
end

@testset "attribution" begin
    # the number of warps uniquely identifies each launch
    p = @retry_busy CUDA.@metrics metrics=["smsp__warps_launched.sum"] begin
        @sync for task in 1:4
            Threads.@spawn for i in 1:4
                @cuda threads=32 blocks=10task+i increment(a)
            end
        end
    end
    @test length(p.kernels.id) == 16
    @test p.values[:, 1] == [grid.x for grid in p.kernels.grid]
end

@testset "contexts" begin
    # only kernels in the profiled context are reported, even if the code switches
    ctx = context()
    CuContext(device()) do other
        context!(ctx)
        p = @retry_busy CUDA.@metrics begin
            @cuda increment(a)
            context!(other)
        end
        @test context() == other
        @test length(p.kernels.id) == 1
        context!(ctx)
    end
end

@testset "replay" begin
    # kernels are replayed to collect metrics, which shouldn't affect results
    fill!(a, 0)
    p = @retry_busy CUDA.@metrics @cuda threads=32 blocks=32 increment(a)
    @test p.passes > 1
    @test all(isequal(1), Array(a))
end

@testset "errors" begin
    executed = Ref(false)
    @test_throws "Metric 'foo__bar.sum' is not available" CUDA.@metrics metrics=["foo__bar.sum"] (executed[] = true)
    @test_throws ArgumentError CUDA.@metrics metrics=:unknown (executed[] = true)
    @test !executed[]

    @test_throws "at most" @retry_busy CUDA.@metrics begin
        for i in 1:CUDA.Profile.max_launches+1
            @cuda increment(a)
        end
    end

    stream = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    graph = capture(; stream) do
        @cuda stream=stream increment(a)
    end
    exec = instantiate(graph)
    @test_throws "CUDA graphs" @retry_busy CUDA.@metrics begin
        launch(exec, stream)
        synchronize(stream)
    end

    @test_throws "profiled code failed" @retry_busy CUDA.@metrics begin
        @cuda increment(a)
        error("profiled code failed")
    end
    @test length(@retry_busy CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
    @test CUDA.@profile(@cuda increment(a)) isa CUDA.Profile.ProfileResults
end

@testset "nesting" begin
    nested(f) = @retry_busy CUDA.@metrics begin
        @cuda increment(a)
        try
            f()
        catch err
            err
        end
    end
    inner_metrics() = CUDA.@metrics metrics=["gpu__time_duration.sum"] @cuda increment(a)
    inner_trace() = CUDA.@profile @cuda increment(a)

    for inner in (inner_metrics, inner_trace)
        # in the same task
        p = nested(inner)
        @test length(p.kernels.id) == 1

        # in another task, which would deadlock if it waited for the outer session
        p = nested(() -> fetch(Threads.@spawn inner()))
        @test length(p.kernels.id) == 1
    end

    # collecting metrics while tracing activity
    str = string(CUDA.@profile begin
        @cuda increment(a)
        try
            inner_metrics()
        catch
        end
    end)
    @test occursin("increment", str)

    @test length(@retry_busy CUDA.@metrics(@cuda increment(a)).kernels.id) == 1
end

end

end

############################################################################################

@testset "sampling" begin

@test_throws ArgumentError @macroexpand CUDA.@sample :flat nothing
@test_throws ArgumentError CUDA.@sample period=4 error("must not execute")
@test_throws ArgumentError CUDA.@sample format=:graph error("must not execute")

# a cubin with a function `kernel`, whose line table has the given sequences of rows
# (address, line, context row, offset of the inlined function's name), and end addresses.
# ptxas relocates the line table with addends for sm_90 and newer, and without (`rel`) before.
function line_table_cubin(sequences; rel::Bool=false)
    uleb(x) = (out = UInt8[]; while true; b = UInt8(x & 0x7f); x >>= 7;
               x == 0 ? (push!(out, b); return out) : push!(out, b | 0x80); end)
    sleb(x) = (out = UInt8[]; while true; b = UInt8(x & 0x7f); x >>= 7;
               done = (x == 0 && b & 0x40 == 0) || (x == -1 && b & 0x40 != 0);
               done ? (push!(out, b); return out) : push!(out, b | 0x80); end)
    bytes(x) = collect(reinterpret(UInt8, [htol(x)]))

    # a DWARF 2 line program, recording where addresses need a relocation
    program = UInt8[]
    relocations = Int[]
    for (rows, stop) in sequences
        append!(program, [0x00, 0x09, 0x02])
        push!(relocations, length(program))
        append!(program, bytes(UInt64(0)))
        address, line, context = 0, 1, 0
        for (row_address, row_line, row_context, func) in rows
            if row_context != context
                operands = vcat(uleb(row_context), uleb(func))
                append!(program, [0x00; uleb(length(operands) + 1); 0x90; operands])
                context = row_context
            end
            append!(program, [0x02; uleb(row_address - address)])
            append!(program, [0x03; sleb(row_line - line)])
            push!(program, 0x01)
            address, line = row_address, row_line
        end
        append!(program, [0x02; uleb(stop - address)])
        append!(program, [0x00, 0x01, 0x01])
    end
    header = UInt8[1, 1, reinterpret(UInt8, Int8(-5)), 14, 13,
                   0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1,     # standard opcode lengths
                   0x00,                                   # no include directories
                   codeunits("test.jl")..., 0, 0, 0, 0,    # one file
                   0x00]
    prefix = 4 + 2 + 4 + length(header)
    debug_line = vcat(bytes(UInt32(prefix - 4 + length(program))), bytes(UInt16(2)),
                      bytes(UInt32(length(header))), header, program)
    relocations .+= prefix

    reloc_section = rel ? ".rel.debug_line" : ".rela.debug_line"
    shstrtab = "\0.shstrtab\0.strtab\0.symtab\0.text.kernel\0.debug_line\0$reloc_section\0.debug_str\0"
    name(section) = findfirst(section * "\0", shstrtab)[1] - 1
    symtab = vcat(zeros(UInt8, 24), bytes(UInt32(1)), UInt8[0x12, 0], bytes(UInt16(4)),
                  bytes(UInt64(0)), bytes(UInt64(0)))
    reloc = reduce(vcat, [vcat(bytes(UInt64(offset)), bytes(UInt64(1) << 32 | 1),
                               rel ? UInt8[] : bytes(Int64(0))) for offset in relocations])
    sections = [(".shstrtab", 3, codeunits(shstrtab), 0, 0),
                (".strtab", 3, codeunits("\0kernel\0"), 0, 0),
                (".symtab", 2, symtab, 2, 1),
                (".text.kernel", 1, zeros(UInt8, 64), 0, 0),
                (".debug_line", 1, debug_line, 0, 0),
                (reloc_section, rel ? 9 : 4, reloc, 3, 5),
                (".debug_str", 1, codeunits("\0inner;\0"), 0, 0)]

    data = zeros(UInt8, 64)
    offsets = Int[]
    for (_, _, contents, _, _) in sections
        push!(offsets, length(data))
        append!(data, contents)
    end
    shoff = length(data)
    append!(data, zeros(UInt8, 64))     # the null section
    for ((section, type, contents, link, info), offset) in zip(sections, offsets)
        append!(data, bytes(UInt32(name(section))), bytes(UInt32(type)), bytes(UInt64(0)),
                bytes(UInt64(0)), bytes(UInt64(offset)), bytes(UInt64(length(contents))),
                bytes(UInt32(link)), bytes(UInt32(info)), bytes(UInt64(1)), bytes(UInt64(0)))
    end
    data[1:6] = [0x7f, UInt8('E'), UInt8('L'), UInt8('F'), 2, 1]
    data[0x29:0x30] = bytes(UInt64(shoff))
    data[0x3b:0x3c] = bytes(UInt16(64))
    data[0x3d:0x3e] = bytes(UInt16(length(sections) + 1))
    data[0x3f:0x40] = bytes(UInt16(1))
    return data
end

@testset "line tables" begin
    lines(table, pcs) = map(pcs) do pc
        chain = CUDATools.source_chain(table, pc)
        isempty(chain) ? nothing : chain[end].line
    end

    # the sequences of a function are merged, and don't extend past their end
    table = CUDATools.line_tables(line_table_cubin([([(0, 1, 0, 0)], 16),
                                                    ([(32, 2, 0, 0)], 48)]))["kernel"]
    @test lines(table, [0, 8, 15, 16, 31, 32, 47, 48, 64]) ==
          [1, 1, 1, nothing, nothing, 2, 2, nothing, nothing]

    # in any order, and when one starts where another ends
    table = CUDATools.line_tables(line_table_cubin([([(16, 2, 0, 0), (24, 3, 0, 0)], 32),
                                                    ([(0, 1, 0, 0)], 16)]))["kernel"]
    @test lines(table, [0, 15, 16, 24, 31, 32]) == [1, 1, 2, 3, 3, nothing]

    # relocations without addends, as used for sm_87 and older
    table = CUDATools.line_tables(line_table_cubin([([(0, 1, 0, 0)], 16)]; rel=true))["kernel"]
    @test lines(table, [0, 15, 16]) == [1, 1, nothing]

    # inlined frames, outermost first, without repeating frames that are inlined at their
    # own location (like Julia's `macro expansion` frames in unrolled code)
    table = CUDATools.line_tables(line_table_cubin([([(0, 10, 0, 0), (8, 20, 1, 1),
                                                      (16, 20, 2, 1)], 24)]))["kernel"]
    outer = CUDATools.SourceFrame("", "test.jl", 10)
    inner = CUDATools.SourceFrame("inner", "test.jl", 20)
    @test CUDATools.source_chain(table, 0) == [outer]
    @test CUDATools.source_chain(table, 8) == [outer, inner]
    @test CUDATools.source_chain(table, 16) == [outer, inner]
    @test isempty(CUDATools.source_chain(table, 24))
end

# a profile of a function with instructions generated for `chains`, sampled `counts` times
function fixture_profile(chains, counts; kwargs...)
    n = length(chains)
    instructions = (; offset=collect(16 .* (0:n-1)), sass=fill("NOP", n), source=chains)
    f = CUDA.Profile.SampledFunction("kernel()", "_Z6kernelv", instructions,
                                     reshape(counts, n, 1), reshape(counts, n, 1))
    CUDA.Profile.PCProfile(; device="GPU", period=14, total=sum(counts), dropped=0,
                             reasons=["wait"], functions=[f], kwargs...)
end

@testset "noise floor" begin
    hot = [CUDATools.SourceFrame("", "kernel.jl", 1)]
    cold = [CUDATools.SourceFrame("", "kernel.jl", 2)]
    p = fixture_profile([hot, cold], [1000, 1])
    f = only(p.functions)
    for format in (:tree, :flat)
        @test length(CUDA.Profile.report(f, format, 0).rows.count) == 2
        @test length(CUDA.Profile.report(f, format, 2).rows.count) == 1
    end
    str = string(p)
    @test occursin("kernel.jl:1  kernel", str)
    @test !occursin("kernel.jl:2", str)

    # when every location is hidden, say so instead of claiming there were no samples
    str = string(fixture_profile([hot], [1]))
    @test occursin("below the noise floor", str)
    @test !occursin("No samples", str)
    @test occursin("kernel.jl:1  kernel", string(fixture_profile([hot], [1]; noisefloor=0)))
end

@testset "display" begin
    kernel = CUDATools.SourceFrame("", "kernel.jl", 1)
    load = CUDATools.SourceFrame("load", "kernel.jl", 10)
    ptr = CUDATools.SourceFrame("pointer", "pointer.jl", 20)
    add = CUDATools.SourceFrame("+", "float.jl", 30)
    p = fixture_profile([[kernel], [kernel, load], [kernel, load, ptr], [kernel, add]],
                        [100, 300, 400, 200]; noisefloor=0)

    # a tree of locations, with the samples, stalled samples and self samples on every row
    lines = split(string(p), '\n')
    @test lines[1] == "kernel()"
    @test lines[2] == "1000 samples, 1000 stalled (100%): execution 100%"
    @test lines[4] == "Samples  Stalled  Self  File:Line  Function  [Stall reasons]"
    @test lines[6:9] == ["   1000     1000   100  kernel.jl:1  kernel  [wait 100%]",
                         "    700      700   300  ├─ kernel.jl:10  load  [wait 100%]",
                         "    400      400   400  │ └─ pointer.jl:20  pointer  [wait 100%]",
                         "    200      200   200  └─ float.jl:30  +  [wait 100%]"]

    flat = string(fixture_profile([[kernel], [kernel, load], [kernel, load, ptr], [kernel, add]],
                                  [100, 300, 400, 200]; noisefloor=0, format=:flat))
    @test occursin("\n    700      700   300  kernel.jl:10  load  [wait 100%]\n", flat)

    # plain output has no escape sequences and isn't cropped; colored output is
    @test !occursin('\e', string(p))
    long = fixture_profile([[CUDATools.SourceFrame("f", "x"^500, 1)]], [1]; noisefloor=0)
    @test !occursin('…', string(long))
    colored = sprint(show, p; context=:color => true)
    @test occursin("\e[", colored)
    # with the same text, and a legend
    stripped = replace(colored, r"\e\[[0-9;]*m" => "")
    @test startswith(stripped, string(p))
    @test occursin("colored by kind", stripped)
    @test !occursin('\e', sprint(CUDATools.code_sass, p))
    @test occursin("\e[", sprint(CUDATools.code_sass, p; context=:color => true))

    # cropped output fits the display
    cropped = sprint(show, MIME"text/plain"(), p; context=(:limit => true, :displaysize => (24, 30)))
    @test all(line -> textwidth(line) <= 30, split(cropped, '\n')[1:9])

    # stall reasons that newer GPUs may add are grouped by their name
    @test CUDA.Profile.stall_kinds[CUDA.Profile.stall_kind("future_throttle")] == "throttle"
    @test CUDA.Profile.stall_kinds[CUDA.Profile.stall_kind("unknown")] == "other"
end

if can_use_cupti() && !CUPTI.has_pc_sampling()
    @test_throws "CUDA 13" CUDA.@sample error("must not execute")
end

# whether this device and driver let us sample PCs
can_sample = can_use_cupti() && CUPTI.has_pc_sampling() && try
    @retry_busy CUDA.@sample nothing
    true
catch err
    msg = sprint(showerror, err)
    contains(msg, "does not support PC sampling") || contains(msg, "performance counters") ||
        contains(msg, "another process") || contains(msg, "stall reasons") || rethrow()
    @warn "Cannot sample PCs, skipping tests" exception=err
    false
end

if can_sample

# a kernel that spends its time on a known line
spin_line = @__LINE__() + 5
function spin(a, n)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    x = a[i]
    for j in 1:n
        x = sqrt(x + Float32(j))
    end
    a[i] = x
    return
end
a = CUDA.ones(Float32, 256 * 64)
spin_kernel() = @cuda threads=256 blocks=64 spin(a, 20_000)
spin_kernel()
synchronize()

is_spin(f) = startswith(f.name, "spin(")
location(frame) = (basename(frame.file), frame.line)

@testset "attribution" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    @test p.total > 0
    f = only(filter(is_spin, p.functions))
    @test sum(f.samples) > 0

    # the kernel spends most of its time on the line in the loop body
    r = CUDA.Profile.report(f, :tree, 0)
    roots = findall(iszero, r.rows.depth)
    hottest = roots[argmax(r.rows.count[roots])]
    @test location(r.rows.path[hottest][end]) == ("profile.jl", spin_line)
    # with `sqrt` inlined into it
    children = [location(path[end]) for (depth, path) in zip(r.rows.depth, r.rows.path)
                if depth == 1 && path[1] == r.rows.path[hottest][1]]
    @test any(((file, _),) -> file == "math.jl", children)

    str = string(p)
    @test occursin("Samples  Stalled  Self  File:Line  Function", str)
    @test occursin(Regex("\\n +\\d+ +\\d+ +\\d+  \\S*profile.jl:$spin_line  spin"), str)
    @test occursin("Total samples: $(p.total)", str)

    flat = string(@retry_busy CUDA.@sample format=:flat spin_kernel())
    @test occursin("Samples  Stalled  Self  File:Line  Function", flat)
    @test occursin("profile.jl:$spin_line  spin", flat)

    # the noise floor hides locations
    rows(noisefloor) = length(CUDA.Profile.report(f, :tree, noisefloor).rows.count)
    @test rows(0) >= rows(2) > 0
    @test rows(1000) == 0

    # frames inlined at their own location (`macro expansion`) aren't repeated in unrolled
    # code, like the loads in a reduction
    x = CUDA.rand(Float32, 2^20)
    sum(x)
    chains = [chain for f in @retry_busy CUDA.@sample(sum(x)).functions
                    for chain in f.instructions.source]
    @test any(chain -> length(chain) > 1, chains)
    for chain in chains
        @test all(i -> chain[i] != chain[i+1], 1:length(chain)-1)
    end

    sass = sprint(CUDATools.code_sass, p)
    @test occursin("/*0000*/", sass)
    @test occursin("MUFU.RSQ", sass)
    @test occursin("profile.jl:$spin_line  spin", sass)
end

@testset "JSON" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    json = JSON.parse(repr("application/json", p))
    @test json["view"] == "tree"
    @test json["total_samples"] == p.total
    @test json["period"] == 14
    @test json["reasons"] == p.reasons
    f = only(filter(f -> startswith(f["name"], "spin("), json["functions"]))
    @test f["count"] == sum(only(filter(is_spin, p.functions)).samples)
    @test sum(sum(values(inst["count"])) for inst in f["instructions"]) == f["count"]
    @test all(f["locations"]) do loc
        loc["frame"] === nothing || 0 <= loc["frame"] < length(json["frames"])
    end
    frame(id) = json["frames"][id+1]
    @test any(f["locations"]) do loc
        loc["depth"] == 0 && loc["frame"] !== nothing &&
            frame(loc["frame"])["line"] == spin_line &&
            endswith(frame(loc["frame"])["file"], "profile.jl")
    end

    json = JSON.parse(repr("application/json", @retry_busy CUDA.@sample format=:flat spin_kernel()))
    @test json["view"] == "flat"
    @test all(loc -> loc["depth"] == 0, only(filter(f -> startswith(f["name"], "spin("),
                                                     json["functions"]))["locations"])
end

@testset "no kernels" begin
    p = @retry_busy CUDA.@sample nothing
    @test isempty(p.functions)
    @test occursin("No samples", string(p))
    @test isempty(JSON.parse(repr("application/json", p))["functions"])
end

@testset "tasks" begin
    p = @retry_busy CUDA.@sample fetch(Threads.@spawn (spin_kernel(); synchronize()))
    @test any(is_spin, p.functions)
end

@testset "errors" begin
    @test_throws "boom" @retry_busy CUDA.@sample error("boom")
    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)

    @test_throws "another profiling session" @retry_busy CUDA.@sample CUDA.@sample nothing
    # `@metrics` rejects unsupported setups before checking for other sessions
    can_collect_metrics &&
        @test_throws "another profiling session" @retry_busy CUDA.@sample CUDA.@metrics nothing
    @test_throws "sampling PCs" @retry_busy CUDA.@sample CUDA.@profile nothing
    @test_throws "another profiling session" CUDA.@profile CUDA.@sample nothing
end

@testset "SASS reflection" begin
    # run `f` in another task, returning its error, or `:timeout` if it deadlocks
    function bounded(f)
        t = Threads.@spawn try
            f()
        catch err
            err
        end
        timedwait(() -> istaskdone(t), 120) == :ok ? fetch(t) : :timeout
    end
    message(err) = err isa Exception ? sprint(showerror, err) : repr(err)

    # reflection that waits for the session would deadlock when the sampled code waits
    # for it, so it's rejected, in the same task or another one
    sass() = CUDATools.code_sass(() -> nothing, devnull)
    device_sass() = @device_code_sass io=devnull nothing
    for inner in (sass, device_sass), nested in (inner, () -> fetch(@async inner()))
        err = bounded(() -> @retry_busy CUDA.@sample nested())
        @test contains(message(err), "Cannot generate SASS code while")
    end

    # reflection already waiting for CUPTI when a session claims it gives up as well. the
    # session's admission can't be paused, so hold the lock and flag the session manually.
    claimed, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = Threads.@spawn lock(CUPTI.callback_lock) do
        put!(claimed, nothing)
        take!(release)
    end
    take!(claimed)
    waiting = Threads.@spawn try
        sass()
    catch err
        err
    end
    sleep(1)
    CUDA.Profile.exclusive_active[] = true
    timedwait(() -> istaskdone(waiting), 120)
    CUDA.Profile.exclusive_active[] = false
    put!(release, nothing)
    wait(holder)
    @test contains(message(fetch(waiting)), "Cannot generate SASS code while")

    # sampling within reflection
    for nested in (() -> CUDA.@sample(nothing), () -> fetch(@async CUDA.@sample(nothing)))
        err = bounded(() -> CUDATools.code_sass(nested, devnull))
        @test contains(message(err), "Cannot sample PCs while CUPTI callbacks are in use")
    end

    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)
    @test CUDATools.code_sass(() -> nothing, devnull) === nothing
end

# CUPTI 13.2+ breaks when loading a module after PC sampling was disabled
@testset "module loading" begin
    p = @retry_busy CUDA.@sample spin_kernel()
    b = CUDA.zeros(Float32, 1)
    p = @retry_busy CUDA.@sample (b .+= 1; synchronize())
    @test p isa CUDA.Profile.PCProfile
    @test CUDA.@profile(b .+= 2) isa CUDA.Profile.ProfileResults
    # whether or not metrics can be collected here, CUPTI shouldn't be broken
    metrics = try
        @retry_busy CUDA.@metrics b .+= 3
    catch err
        err
    end
    @test !occursin("INVALID_OPERATION", sprint(show, metrics))
    @test any(is_spin, @retry_busy CUDA.@sample(spin_kernel()).functions)
end

end

end

end

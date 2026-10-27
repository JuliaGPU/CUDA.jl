# PC sampling

using ..CUDATools: SourceFrame, LineTable, line_tables, source_chain, ELFFile, symbol_index,
                   ptx_text
import ..CUDATools: code_sass, code_ptx

"""
    SampledFunction

The samples collected in a GPU function (a kernel, and any non-inlined functions it calls)
by [`CUDA.@sample`](@ref CUDATools.@sample), with the following fields:

- `name`: the (demangled) name of the function, and `symbol` its mangled name;
- `instructions`: a table (NamedTuple of vectors) with the function's SASS instructions:
  their `offset` in the function, the instruction text (`sass`, or labels with offset -1),
  and the `source` locations they were generated from (outermost first, including inlined
  frames; empty if unknown). If the code couldn't be disassembled, this only lists the
  sampled instructions, without text;
- `samples` and `stalled`: matrices with, for every instruction (row), the number of
  samples per reason (column), counting all samples, or only the samples where the warp
  scheduler didn't issue an instruction (i.e., where stalls weren't hidden by other
  warps);
- `ptx`: the PTX code of the module, as ptxas embedded it in the cubin (`text`, a vector of
  lines, with directives like `.loc` blanked out), and the line of that code every
  instruction was generated from (`lines`, 0 if unknown), or `nothing` if the code was
  compiled without line information.

See [`CUDA.@sample`](@ref CUDATools.@sample) for the ways to display these samples.
"""
struct SampledFunction
    name::String
    symbol::String
    instructions::@NamedTuple{offset::Vector{Int}, sass::Vector{String},
                              source::Vector{Vector{SourceFrame}}}
    samples::Matrix{Int}
    stalled::Matrix{Int}
    ptx::Union{Nothing,@NamedTuple{text::Vector{String}, lines::Vector{Int}}}
end

"""
    PCProfile

The samples collected by [`CUDA.@sample`](@ref CUDATools.@sample), which describes the ways
to display them.

The text report shows, for every sampled function, a tree of source locations where the
children of a location are the functions that were inlined into it, or a flat list of
locations. Every location lists the number of samples of the instructions generated for it
(`Samples`), how many of them were `Stalled`, the samples of its own code, excluding
inlined functions (`Self`), and the most common reasons for the stalls. Stall reasons are
grouped into kinds (`memory`, `throttle`, `sync`, `execution` and `issue`).

`show(io, MIME"application/json"(), p)` writes the samples as JSON:
- `view`: `"tree"` or `"flat"`, depending on the format;
- `device`, the sampling `period` (once every 2^period cycles), the `noisefloor`, and the
  number of `total_samples` (including samples outside of the sampled functions) and
  `dropped_samples`;
- `reasons`: the reasons warps were stalled for;
- `frames`: source locations, each with a `function` (empty for the sampled function
  itself), `file` and `line`;
- `ptx`: the PTX code of the sampled modules, as embedded in their cubins (see
  [`CUDATools.code_ptx`](@ref CUDATools.code_ptx(::IO, ::CUDATools.Profile.PCProfile)));
- `functions`: every sampled function with its demangled `name` and mangled `symbol`, its
  `count` of samples and how many of them were `stalled`, its `ptx` code (by index, from 0,
  or `null` without line information), the `locations` that are displayed (in order, each
  with its `depth` in the tree, its `frame` (by index, from 0, or `null` if unknown),
  `count`, `stalled` and `overhead` samples, and stalled samples per reason), and every
  sampled instruction with its `offset`, `sass` text, the `frames` it was generated from
  (outermost first), the line of PTX code it was generated from (`ptx_line`, from 1, or
  `null` if unknown), and its `count` and `stalled` samples per reason. Reasons without
  samples are omitted.

The following fields are also available:
- `device`, `period`, `noisefloor`, `total` and `dropped`, as above, and the `format`;
- `reasons`: the reasons that index the columns of the sample matrices;
- `functions`: a vector of [`Profile.SampledFunction`](@ref CUDATools.Profile.SampledFunction),
  sorted by their number of stalled samples.
"""
Base.@kwdef struct PCProfile
    device::String
    period::Int
    total::Int
    dropped::Int
    reasons::Vector{String}
    functions::Vector{SampledFunction}

    # display properties set by `@sample` kwargs
    format::Symbol=:tree
    noisefloor::Float64=2.0
end


## collection

function check_sampling_support()
    if !CUPTI.can_profile()
        error("""Sampling PCs relies on CUPTI, which requires additional permissions on Tegra devices.
                 See the documentation of `CUDA.@profile` for details.""")
    end
    if !CUPTI.has_pc_sampling()
        error("Sampling PCs requires CUPTI from CUDA 13 or later (CUPTI $(CUPTI.version()) is loaded).")
    end
    if detect_cupti()
        error("This Julia session is already being profiled by an external tool, so CUDA.jl cannot sample PCs.")
    end
end

function sample_pcs(@nospecialize(f); period::Integer=14, format::Symbol=:tree,
                    noisefloor::Real=2.0)
    5 <= period <= 31 ||
        throw(ArgumentError("The sampling period must be between 5 and 31 (2^5 to 2^31 cycles)."))
    format in (:tree, :flat) ||
        throw(ArgumentError("Invalid format `:$format`; use `:tree` or `:flat`."))
    check_sampling_support()

    # claim CUPTI without waiting, as the code being sampled may be waiting for whoever
    # holds it. besides the activity API, we use callbacks to capture cubins.
    if islocked(CUPTI.activity_lock) || !trylock(CUPTI.activity_lock)
        error("Cannot sample PCs while another profiling session is active.")
    end
    try
        if islocked(CUPTI.callback_lock) || !trylock(CUPTI.callback_lock)
            error("Cannot sample PCs while CUPTI callbacks are in use (e.g. by `@device_code_sass`).")
        end
        exclusive_active[] = true
        try
            collect_samples(f, period, format, noisefloor)
        finally
            exclusive_active[] = false
            unlock(CUPTI.callback_lock)
        end
    finally
        unlock(CUPTI.activity_lock)
    end
end

function collect_samples(@nospecialize(f), period, format, noisefloor)
    ctx = context()

    # PC samples identify the code by the CRC of its cubin, so capture the cubins that are
    # executed. the callback runs while the driver holds locks, so only copy them. it can
    # run concurrently on several threads, but must not yield, so guard with a spin lock.
    cubins = Dict{UInt32,Vector{UInt8}}()
    cubins_lock = Threads.SpinLock()
    cfg = CUPTI.CallbackConfig([CUPTI.CUPTI_CB_DOMAIN_RESOURCE]) do domain, id, data
        id == CUPTI.CUPTI_CBID_RESOURCE_MODULE_PROFILED || return
        desc = unsafe_load(convert(Ptr{CUPTI.CUpti_ModuleResourceData},
                                   data.resourceDescriptor))
        seen = @lock cubins_lock haskey(cubins, desc.moduleId)
        seen && return
        cubin = copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(pointer(desc.pCubin)), desc.cubinSize))
        @lock cubins_lock get!(cubins, desc.moduleId, cubin)
        return
    end

    CUDACore.cuCtxSynchronize()
    samples = try
        try
            CUPTI.@enable! cfg CUPTI.pc_sample(; ctx, period) do
                f()
                # the code may have left another context current
                context!(CUDACore.cuCtxSynchronize, ctx)
            end
        finally
            # work around an NVIDIA bug that breaks later profiling (see `CUPTI.pc_sample`)
            # by detaching CUPTI, which we're the only user of while holding its locks.
            CUPTI.cuptiFinalize()
        end
    catch err
        if err isa CUPTIError && err.code == CUPTI.CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
            error("""Sampling PCs requires access to the GPU's performance counters,
                     which the NVIDIA driver restricts to administrators by default. Either run Julia as
                     root, or allow all users by loading the `nvidia` kernel module with
                     `NVreg_RestrictProfilingToAdminUsers=0`.
                     See https://developer.nvidia.com/ERR_NVGPUCTRPERM for details.""")
        elseif err isa CUPTIError && err.code == CUPTI.CUPTI_ERROR_HARDWARE_BUSY
            error("Cannot sample PCs, as the GPU is being profiled by another process.")
        end
        rethrow()
    end

    PCProfile(; device=name(device(ctx)), period, total=samples.total,
                dropped=samples.dropped, format, noisefloor,
                sampled_functions(samples, values(cubins))...)
end

const reason_prefix = "smsp__pcsamp_warps_issue_stalled_"

# attribute PC samples to the instructions and source lines of the sampled functions
function sampled_functions(samples::CUPTI.PCSamples, cubins)
    # CUPTI reports every stall reason twice: counting all samples, and only those where
    # no instruction was issued. it also reports the number of samples as a reason.
    reasons = String[]
    columns = Dict{UInt32,Tuple{Int,Bool}}()     # CUPTI index => (column, stalled)
    for (index, name) in sort!(collect(samples.reasons); by=last)
        startswith(name, reason_prefix) || continue
        reason = chopprefix(name, reason_prefix)
        stalled = endswith(reason, "_not_issued")
        reason = chopsuffix(reason, "_not_issued")
        column = findfirst(==(reason), reasons)
        if column === nothing
            push!(reasons, reason)
            column = length(reasons)
        end
        columns[index] = (column, stalled)
    end

    by_crc = Dict(CUPTI.cubin_crc(cubin) => cubin for cubin in cubins)
    pcs = Dict{Tuple{UInt64,String},Vector{Tuple{Int,Dict{UInt32,Int}}}}()
    for ((crc, symbol, offset), counts) in samples.pcs
        push!(get!(Vector{Tuple{Int,Dict{UInt32,Int}}}, pcs, (crc, symbol)),
              (Int(offset), counts))
    end

    tables = Dict{UInt64,Dict{String,LineTable}}()
    ptx_code = Dict{UInt64,Union{Nothing,Tuple{Vector{String},Dict{String,LineTable}}}}()
    functions = SampledFunction[]
    for ((crc, symbol), records) in pcs
        cubin = get(by_crc, crc, nothing)
        table = cubin === nothing ? nothing : get(get!(tables, crc) do
            try
                line_tables(cubin)
            catch err
                @debug "Could not decode the line table of a cubin" exception=(err, catch_backtrace())
                Dict{String,LineTable}()
            end
        end, symbol, nothing)
        sass = cubin === nothing ? nothing : disassemble_function(cubin, symbol)
        ptx = cubin === nothing ? nothing : get!(() -> embedded_ptx(cubin), ptx_code, crc)

        # the instructions, or only the sampled ones if we couldn't disassemble the code
        offsets, texts = if sass === nothing
            sampled = sort!(Int[offset for (offset, _) in records])
            sampled, fill("", length(sampled))
        else
            sass
        end
        source = Vector{SourceFrame}[offset < 0 || table === nothing ? SourceFrame[] :
                                     source_chain(table, offset) for offset in offsets]
        rows = Dict(offset => i for (i, offset) in enumerate(offsets) if offset >= 0)
        all_samples = zeros(Int, length(offsets), length(reasons))
        stalled = zeros(Int, length(offsets), length(reasons))
        for (offset, counts) in records
            row = get(rows, offset, nothing)
            row === nothing && continue
            for (index, count) in counts
                column, is_stalled = get(columns, index, (0, false))
                column == 0 && continue
                (is_stalled ? stalled : all_samples)[row, column] += count
            end
        end
        instructions = (; offset=offsets, sass=texts, source)
        ptx_table = ptx === nothing ? nothing : get(ptx[2], symbol, nothing)
        ptx_lines = if ptx_table === nothing
            nothing
        else
            lines = map(offsets) do offset
                chain = offset < 0 ? SourceFrame[] : source_chain(ptx_table, offset)
                isempty(chain) ? 0 : chain[end].line
            end
            (; text=ptx[1], lines)
        end
        push!(functions, SampledFunction(symbol, symbol, instructions, all_samples, stalled,
                                         ptx_lines))
    end

    names = demangle_names!([f.symbol for f in functions])
    functions = [SampledFunction(name, f.symbol, f.instructions, f.samples, f.stalled, f.ptx)
                 for (name, f) in zip(names, functions)]
    sort!(functions; by=f -> (sum(f.stalled), sum(f.samples)), rev=true)
    return (; reasons, functions)
end

# the PTX code embedded in `cubin`, as lines, and the line tables that map the instructions
# of its functions onto them. this is the data Nsight Compute uses to correlate SASS and PTX.
function embedded_ptx(cubin::Vector{UInt8})
    try
        text = ptx_text(cubin)
        text === nothing && return nothing
        return String.(split(text, '\n')), line_tables(cubin; section=".nv_debug_line_sass")
    catch err
        @debug "Could not read the PTX code of a cubin" exception=(err, catch_backtrace())
        return nothing
    end
end

# the SASS instructions of `symbol` in `cubin`, as (offsets, texts), with labels at offset -1
function disassemble_function(cubin::Vector{UInt8}, symbol::String)
    index = try
        symbol_index(ELFFile(cubin), symbol)
    catch err
        @debug "Could not read the symbols of a cubin" exception=(err, catch_backtrace())
        nothing
    end
    index === nothing && return nothing

    lines = try
        mktemp() do path, io
            write(io, cubin)
            close(io)
            readlines(`$(CUDACore.CUDA_Compiler.nvdisasm()::Cmd) --print-code -fun $index $path`)
        end
    catch err
        @debug "Could not disassemble a cubin" exception=(err, catch_backtrace())
        return nothing
    end

    offsets = Int[]
    texts = String[]
    for line in lines
        m = match(r"^\s*/\*([0-9a-f]+)\*/\s+(.*?)\s*;?\s*$", line)
        if m !== nothing
            push!(offsets, parse(Int, m[1]; base=16))
            push!(texts, m[2])
        elseif occursin(r"^\.L\w*:$", line)
            push!(offsets, -1)
            push!(texts, line)
        end
    end
    return offsets, texts
end


## report

# The report is modeled on `Profile.print`: a tree of source locations per sampled
# function, where the children of a location are the functions inlined into it, or a flat
# list of locations. Every location counts the samples of the instructions generated for it
# (including inlined code), the samples where warps were stalled, and the samples of
# instructions that weren't inlined further (the overhead).

mutable struct SourceNode
    frame::SourceFrame
    count::Int
    stalled::Int
    overhead::Int
    reasons::Vector{Int}            # stalled samples per reason
    children::Dict{SourceFrame,SourceNode}
end
SourceNode(frame::SourceFrame, nreasons::Int) =
    SourceNode(frame, 0, 0, 0, zeros(Int, nreasons), Dict{SourceFrame,SourceNode}())

const unknown_frame = SourceFrame("", "", 0)

function add_samples!(node::SourceNode, f::SampledFunction, i::Int)
    node.count += sum(@view f.samples[i, :])
    node.stalled += sum(@view f.stalled[i, :])
    node.reasons .+= @view f.stalled[i, :]
end

function source_tree(f::SampledFunction)
    root = SourceNode(unknown_frame, size(f.samples, 2))
    for i in axes(f.samples, 1)
        any(!iszero, @view f.samples[i, :]) || continue
        chain = f.instructions.source[i]
        node = root
        add_samples!(node, f, i)
        for frame in (isempty(chain) ? [unknown_frame] : chain)
            node = get!(node.children, frame) do
                SourceNode(frame, size(f.samples, 2))
            end
            add_samples!(node, f, i)
        end
        node.overhead += sum(@view f.samples[i, :])
    end
    return root
end

# order locations by how much warps were stalled there, which is time spent waiting
order(node::SourceNode) = (node.stalled, node.count)

# show a location only if its count exceeds the noise floor, relative to its parent
above_noise(count, parent, noisefloor) = count > noisefloor * sqrt(parent)

# the rows that displaying `f` shows, in order
report(f::SampledFunction, format::Symbol, noisefloor::Real) =
    format == :flat ? flat_report(f, noisefloor) : tree_report(f, noisefloor)

function tree_report(f::SampledFunction, noisefloor::Real)
    root = source_tree(f)
    rows = (; depth=Int[], path=Vector{SourceFrame}[], count=Int[], stalled=Int[],
              overhead=Int[], reasons=Vector{Int}[])
    tree_rows!(rows, root, 0, SourceFrame[], noisefloor)
    return (; rows, count=root.count, stalled=root.stalled)
end

# add the rows for the children of `node` above the noise floor, depth-first
function tree_rows!(rows, node, depth, path, noisefloor)
    for child in sort!(collect(values(node.children)); by=order, rev=true)
        above_noise(child.count, node.count, noisefloor) || continue
        child_path = vcat(path, child.frame)
        push!(rows.depth, depth)
        push!(rows.path, child_path)
        push!(rows.count, child.count)
        push!(rows.stalled, child.stalled)
        push!(rows.overhead, child.overhead)
        push!(rows.reasons, child.reasons)
        tree_rows!(rows, child, depth + 1, child_path, noisefloor)
    end
end

function flat_report(f::SampledFunction, noisefloor::Real)
    nodes = Dict{SourceFrame,SourceNode}()
    total = SourceNode(unknown_frame, size(f.samples, 2))
    for i in axes(f.samples, 1)
        any(!iszero, @view f.samples[i, :]) || continue
        add_samples!(total, f, i)
        chain = f.instructions.source[i]
        chain = isempty(chain) ? [unknown_frame] : chain
        # count every location once, even if it occurs several times in the chain
        for frame in unique(chain)
            node = get!(() -> SourceNode(frame, size(f.samples, 2)), nodes, frame)
            add_samples!(node, f, i)
        end
        nodes[chain[end]].overhead += sum(@view f.samples[i, :])
    end
    selected = filter(node -> above_noise(node.count, total.count, noisefloor),
                      collect(values(nodes)))
    sort!(selected; by=order, rev=true)
    rows = (; depth=zeros(Int, length(selected)),
              path=Vector{SourceFrame}[[node.frame] for node in selected],
              count=Int[node.count for node in selected],
              stalled=Int[node.stalled for node in selected],
              overhead=Int[node.overhead for node in selected],
              reasons=Vector{Int}[node.reasons for node in selected])
    return (; rows, count=total.count, stalled=total.stalled)
end

# like `Profile.print`, abbreviate the paths of Base and of packages as `@Name/path`, and
# show other files relative to the working directory if possible
function format_path(file::String)
    startswith(file, "./") && return "@Base/" * file[3:end]
    m = match(r"/packages/([^/]+)/[^/]+/(.*)$", file)
    m === nothing || return "@$(m[1])/$(m[2])"
    dir = try
        pwd() * "/"
    catch
        nothing     # the working directory was removed
    end
    dir !== nothing && startswith(file, dir) && return file[length(dir)+1:end]
    m = match(r"^(.*)/src/", file)
    if m !== nothing
        name = package_name(String(m[1]))
        name === nothing || return "@$name/" * file[length(m[1])+2:end]
    end
    return Base.contractuser(file)
end

# the name of the package at `dir`, if any
const package_names = Dict{String,Union{Nothing,String}}()
package_name(dir::String) = get!(package_names, dir) do
    project = joinpath(dir, "Project.toml")
    isfile(project) || return nothing
    for line in eachline(project)
        m = match(r"^name\s*=\s*\"(.+)\"", line)
        m === nothing || return String(m[1])
    end
    return nothing
end

# the name of a function, without its arguments
function short_name(name::String)
    i = findfirst('(', name)
    i === nothing ? name : name[1:prevind(name, i)]
end

# CUPTI's stall reasons, grouped by what warps are waiting for (following the warp scheduler
# states that Nsight Compute documents): results from memory or other variable-latency
# units, a full queue or busy pipeline, synchronization, the fixed latency of an earlier
# instruction, or the scheduler and instruction fetch. Reasons that aren't listed, e.g. on
# newer GPUs, are grouped by their name.
const stall_kinds = ["memory", "throttle", "sync", "execution", "issue", "other"]
const stall_kind_of = Dict(
    "long_scoreboard" => 1, "short_scoreboard" => 1, "imc_miss" => 1, "drain" => 1,
    "lg_throttle" => 2, "mio_throttle" => 2, "tex_throttle" => 2, "math_pipe_throttle" => 2,
    "barrier" => 3, "membar" => 3, "sleeping" => 3,
    "wait" => 4,
    "no_instructions" => 5, "branch_resolving" => 5, "dispatch_stall" => 5,
    "not_selected" => 5, "selected" => 5)
function stall_kind(reason::String)
    kind = get(stall_kind_of, reason, 0)
    kind != 0 && return kind
    endswith(reason, "_throttle") && return 2
    endswith(reason, "_scoreboard") && return 1
    return length(stall_kinds)
end

# stalled samples per kind, given the stalled samples per reason
function kind_counts(reasons::Vector{String}, counts::AbstractVector{Int})
    kinds = zeros(Int, length(stall_kinds))
    for (reason, count) in zip(reasons, counts)
        kinds[stall_kind(reason)] += count
    end
    kinds
end

# ANSI styles, only written when the output supports color: stalls in a color per kind,
# hot code in bold, and context (paths, tree guides, zeros) dimmed
const kind_styles = ["\e[31m", "\e[33m", "\e[35m", "\e[94m", "\e[36m", ""]
const bold_style = "\e[1m"
const dim_style = "\e[90m"

# code with at least this share of a function's samples is hot
const hot_share = 0.1
is_hot(count::Int, total::Int) = count > 0 && count >= hot_share * total

# a line of output as segments of text and their style, so that it can be cropped without
# breaking escape sequences
const Line = Vector{Tuple{String,String}}

# like `Profile.print`, crop lines that don't fit the display
function print_line(io::IO, line::Line, width::Int)
    color = get(io, :color, false)::Bool
    cropped = sum(segment -> textwidth(segment[1]), line; init=0) > width
    available = cropped ? max(width - 1, 0) : typemax(Int)
    for (text, style) in line
        available <= 0 && break
        isempty(text) && continue
        if textwidth(text) > available
            n = 0
            for (i, c) in pairs(text)
                n + textwidth(c) > available && (text = text[1:prevind(text, i)]; break)
                n += textwidth(c)
            end
        end
        available -= textwidth(text)
        if color && !isempty(style)
            print(io, style, text, "\e[0m")
        else
            print(io, text)
        end
    end
    cropped && print(io, "…")
    println(io)
end

function push_location!(line::Line, frame::SourceFrame, kernel::String, hot::Bool)
    if frame === unknown_frame
        push!(line, ("?  (unknown location)", dim_style))
    else
        push!(line, ("$(format_path(frame.file)):$(frame.line)", dim_style))
        push!(line, ("  ", ""))
        push!(line, (isempty(frame.func) ? kernel : frame.func, hot ? bold_style : ""))
    end
    line
end

# the reasons that make up most of the stalled samples, as (reason, percentage)
function top_reasons(counts::AbstractVector{Int}, n::Int)
    total = sum(counts)
    top = Tuple{Int,Int}[]
    total == 0 && return top
    for i in sortperm(counts; rev=true)[1:min(n, end)]
        share = percentage(counts[i], total)
        share == 0 && break
        push!(top, (i, share))
    end
    top
end

function push_reasons!(line::Line, reasons::Vector{String}, counts::AbstractVector{Int})
    top = top_reasons(counts, 3)
    isempty(top) && return line
    push!(line, ("  [", ""))
    for (k, (i, share)) in enumerate(top)
        k > 1 && push!(line, (", ", ""))
        push!(line, (reasons[i], kind_styles[stall_kind(reasons[i])]))
        push!(line, (" $share%", ""))
    end
    push!(line, ("]", ""))
end

# the number of samples of a function, how many were stalled, and the stall kinds
function summary_line(count::Int, stalled::Int, reasons::Vector{String},
                      counts::AbstractVector{Int})
    line = Line([("$count samples, $stalled stalled ($(percentage(stalled, count))%)", "")])
    kinds = kind_counts(reasons, counts)
    first = true
    for k in sortperm(kinds; rev=true)
        share = percentage(kinds[k], stalled)
        share == 0 && break
        push!(line, (first ? ": " : ", ", ""))
        push!(line, (stall_kinds[k], kind_styles[k]))
        push!(line, (" $share%", ""))
        first = false
    end
    line
end

# the style of a number of stalled samples: the color of their most common kind
function stalled_style(reasons::Vector{String}, counts::AbstractVector{Int})
    kinds = kind_counts(reasons, counts)
    all(iszero, kinds) ? dim_style : kind_styles[argmax(kinds)]
end

percentage(x::Int, total::Int) = total == 0 ? 0 : round(Int, 100 * x / total)

Base.show(io::IO, p::PCProfile) = show_report(io, p)

function show_results(io::IOContext{IOBuffer}, p::PCProfile, crop::Symbol)
    width = crop == :horizontal ? displaysize(io)[2] : typemax(Int)
    shown = 0
    hidden = (; functions=0, count=0)
    for f in p.functions
        r = barrier(report, f, p.format, p.noisefloor)
        if isempty(r.rows.count)
            r.count > 0 && (hidden = (; functions=hidden.functions+1, count=hidden.count+r.count))
            continue
        end
        barrier(show_function, io, f, r, p.reasons, p.format, width)
        shown += 1
    end
    if shown == 0 && hidden.count > 0
        println(io, "All $(hidden.count) samples in GPU code are below the noise floor. Use a lower `noisefloor` (e.g. `CUDA.@sample noisefloor=0 ...`) to show them.\n")
    elseif shown == 0
        println(io, "No samples were collected in GPU code. Make sure the code runs long enough, or use a shorter sampling period.\n")
    elseif hidden.functions > 0
        println(io, "$(hidden.functions) more function$(hidden.functions == 1 ? "" : "s") with $(hidden.count) samples.\n")
    end

    print(io, "Total samples: $(p.total)")
    p.dropped > 0 && print(io, " ($(p.dropped) dropped)")
    println(io, ", sampling every 2^$(p.period) cycles per SM on $(p.device).")
    println(io, """
        Stalled samples are those where the warp's scheduler issued no instruction, which
        other warps didn't hide; stall reasons are shares of them. Self samples are those of
        the location's own code, excluding inlined functions. Inlined code is attributed by
        ptxas, which can be inaccurate. Use `CUDATools.code_sass(p)` or `CUDATools.code_ptx(p)`
        to see the samples per instruction.""")
    get(io, :color, false)::Bool && show_legend(io)
end

function show_legend(io::IO)
    line = Line([("Stalls are colored by kind: ", "")])
    for (k, kind) in enumerate(stall_kinds[1:end-1])
        k > 1 && push!(line, (", ", ""))
        push!(line, (kind, kind_styles[k]))
    end
    push!(line, ("; code with at least $(round(Int, 100 * hot_share))% of the samples is ", ""))
    push!(line, ("bold", bold_style))
    push!(line, (".", ""))
    print_line(io, line, typemax(Int))
end

function show_function(io::IOContext{IOBuffer}, f::SampledFunction, r::NamedTuple,
                       reasons::Vector{String}, format::Symbol, width::Int)
    kernel = short_name(f.name)
    print_line(io, Line([(f.name, bold_style)]), width)
    print_line(io, summary_line(r.count, r.stalled, reasons, vec(sum(f.stalled; dims=1))),
               width)
    println(io)

    rows = r.rows
    widths = (max(7, ndigits(maximum(rows.count))), max(7, ndigits(maximum(rows.stalled))),
              max(4, ndigits(maximum(rows.overhead))))
    header = string(lpad("Samples", widths[1]), "  ", lpad("Stalled", widths[2]), "  ",
                    lpad("Self", widths[3]), "  File:Line  Function  [Stall reasons]")
    print_line(io, Line([(header, bold_style)]), width)
    print_line(io, Line([("─"^textwidth(header), dim_style)]), width)
    guides = format == :flat ? fill("", length(rows.depth)) : tree_guides(rows.depth)
    for i in eachindex(rows.count)
        hot = is_hot(rows.overhead[i], r.count)
        line = Line([(lpad(rows.count[i], widths[1]), ""), ("  ", ""),
                     (lpad(rows.stalled[i], widths[2]), stalled_style(reasons, rows.reasons[i])),
                     ("  ", ""),
                     (lpad(rows.overhead[i], widths[3]),
                      hot ? bold_style : rows.overhead[i] == 0 ? dim_style : ""),
                     ("  ", ""), (guides[i], dim_style)])
        push_location!(line, rows.path[i][end], kernel, hot)
        push_reasons!(line, reasons, rows.reasons[i])
        print_line(io, line, width)
    end
    println(io)
end

# the guides that draw a tree with rows at the given depths, two columns per level so that
# deeply inlined code stays readable
function tree_guides(depths::Vector{Int})
    guides = Vector{String}(undef, length(depths))
    siblings = Bool[]   # per level, whether a later row continues it
    for i in reverse(eachindex(depths))
        depth = depths[i]
        while length(siblings) <= depth
            push!(siblings, false)
        end
        guide = IOBuffer()
        for level in 1:depth-1
            print(guide, siblings[level+1] ? "│ " : "  ")
        end
        depth > 0 && print(guide, siblings[depth+1] ? "├─ " : "└─ ")
        guides[i] = String(take!(guide))
        siblings[depth+1] = true
        resize!(siblings, depth + 1)
    end
    guides
end


## SASS and PTX code

"""
    code_sass([io=stdout], p::Profile.PCProfile; debuginfo=:default)

Print the SASS code of the functions sampled by [`CUDA.@sample`](@ref CUDATools.@sample),
annotating every instruction with its number of samples, of stalled samples, and its most
common stall reason. Unless `debuginfo=:none`, the source locations the instructions were
generated from are shown as comments, including the frames of inlined functions, like
[`@device_code_sass`](@ref CUDATools.@device_code_sass) does. Functions without samples
are omitted.

See also:
[`CUDATools.code_ptx`](@ref CUDATools.code_ptx(::IO, ::CUDATools.Profile.PCProfile))
"""
function code_sass(io::IO, p::PCProfile; debuginfo::Symbol=:default)
    show_code(io, p) do buf, f
        barrier(show_sass, buf, f, p.reasons, debuginfo != :none)
    end
end
code_sass(p::PCProfile; kwargs...) = code_sass(stdout, p; kwargs...)

"""
    code_ptx([io=stdout], p::Profile.PCProfile; raw=false, debuginfo=:default)

Print the PTX code of the functions sampled by [`CUDA.@sample`](@ref CUDATools.@sample),
annotating every line with the samples of the SASS instructions that ptxas generated for
it, as recorded in the cubin. This includes the code of functions that ptxas inlined.
Unless `debuginfo=:none`, source locations are shown as comments, taken from the first
instruction generated for a line that has one. With `raw=true`, the code is printed as ptxas embedded it
in the cubin, with an empty line for every directive (like `.loc`) and with the labels
used by debug information.

This requires line information, i.e., running Julia with a debug level of at least 1 (the
default).

See also:
[`CUDATools.code_sass`](@ref CUDATools.code_sass(::IO, ::CUDATools.Profile.PCProfile))
"""
function code_ptx(io::IO, p::PCProfile; raw::Bool=false, debuginfo::Symbol=:default)
    show_code(io, p) do buf, f
        barrier(show_ptx, buf, f, p.reasons, raw, debuginfo != :none)
    end
end
code_ptx(p::PCProfile; kwargs...) = code_ptx(stdout, p; kwargs...)

function show_code(f, io::IO, p::PCProfile)
    buf = IOContext(IOBuffer(), io)
    for fn in p.functions
        sum(fn.samples) == 0 && continue
        f(buf, fn)
    end
    write(io, take!(buf.io))
    return
end

# the most common reason is shown as `reason 100%`, followed by a space
reason_width(reasons::Vector{String}) = 6 + maximum(length, reasons; init=0)
margin_width(reasons::Vector{String}) = 10 + 2 + 7 + 2 + reason_width(reasons)

# the columns with the samples of a line of code
function sample_columns(reasons::Vector{String}, samples::AbstractVector{Int},
                        stalled::AbstractVector{Int}, hot::Bool)
    count = sum(samples)
    count == 0 && return Line([(" "^margin_width(reasons), "")])
    width = reason_width(reasons)
    top = top_reasons(stalled, 1)
    reason = isempty(top) ? Line([(" "^width, "")]) :
        Line([(reasons[top[1][1]], kind_styles[stall_kind(reasons[top[1][1]])]),
              (rpad(" $(top[1][2])%", width - length(reasons[top[1][1]])), "")])
    Line([(lpad(count, 10), hot ? bold_style : ""), ("  ", ""),
          (lpad(sum(stalled), 7), stalled_style(reasons, stalled)), ("  ", ""),
          reason...])
end

function show_code_header(io::IO, f::SampledFunction, reasons::Vector{String};
                          note::Union{Nothing,String}=nothing)
    print_line(io, Line([("// ", ""), (f.name, bold_style)]), typemax(Int))
    print_line(io, pushfirst!(summary_line(sum(f.samples), sum(f.stalled), reasons,
                                           vec(sum(f.stalled; dims=1))), ("// ", "")),
               typemax(Int))
    note === nothing || println(io, "// ", note)
    println(io, "// Samples  Stalled  Stall reason")
end

# print the frames of the source location `chain` that are entered when coming from the
# location `stack`, nesting inlined code
function print_frames(io::IO, stack::Vector{SourceFrame}, chain::Vector{SourceFrame},
                      margin::Int, comment::String, kernel::String)
    common = 0
    while common < min(length(stack), length(chain)) && stack[common+1] == chain[common+1]
        common += 1
    end
    for depth in common+1:length(chain)
        line = Line([(" "^margin * comment * " "^(depth-1), "")])
        print_line(io, push_location!(line, chain[depth], kernel, false), typemax(Int))
    end
end

function show_sass(io::IOContext{IOBuffer}, f::SampledFunction, reasons::Vector{String},
                   frames::Bool)
    show_code_header(io, f, reasons)
    kernel = short_name(f.name)
    total = sum(f.samples)
    margin = margin_width(reasons)
    stack = SourceFrame[]
    for i in eachindex(f.instructions.offset)
        offset = f.instructions.offset[i]
        if offset < 0
            println(io, f.instructions.sass[i])
            continue
        end

        chain = f.instructions.source[i]
        if frames
            print_frames(io, stack, chain, margin, "; ", kernel)
            stack = chain
        end

        hot = is_hot(sum(@view f.samples[i, :]), total)
        line = sample_columns(reasons, @view(f.samples[i, :]), @view(f.stalled[i, :]), hot)
        push!(line, ("/*$(string(offset; base=16, pad=4))*/ ", ""))
        push!(line, (f.instructions.sass[i], hot ? bold_style : ""))
        print_line(io, line, typemax(Int))
    end
    println(io)
end

function show_ptx(io::IOContext{IOBuffer}, f::SampledFunction, reasons::Vector{String},
                  raw::Bool, frames::Bool)
    if f.ptx === nothing
        print_line(io, Line([("// ", ""), (f.name, bold_style)]), typemax(Int))
        println(io, "// No PTX code available, which requires line information.")
        println(io)
        return
    end
    text, ptx_lines = f.ptx

    # the samples of every line, and the source location of its first SASS instruction.
    # ptxas generates all instructions of a line from the same `.loc`, so they should have
    # the same location, but the line table isn't always consistent.
    nreasons = length(reasons)
    counts = Dict{Int,Tuple{Vector{Int},Vector{Int}}}()
    chains = Dict{Int,Vector{SourceFrame}}()
    unknown = 0
    for i in eachindex(f.instructions.offset)
        f.instructions.offset[i] < 0 && continue
        line = ptx_lines[i]
        if line == 0
            unknown += sum(@view f.samples[i, :])
            continue
        end
        samples, stalled = get!(() -> (zeros(Int, nreasons), zeros(Int, nreasons)),
                                counts, line)
        samples .+= @view f.samples[i, :]
        stalled .+= @view f.stalled[i, :]
        chain = f.instructions.source[i]
        isempty(chain) || get!(chains, line, chain)
    end
    total = sum(f.samples)
    note = unknown == 0 ? nothing :
        "$unknown samples ($(percentage(unknown, total))%) are of instructions without a " *
        "line of PTX code"
    show_code_header(io, f, reasons; note)

    # the function, and the functions that ptxas inlined into it
    kernel = short_name(f.name)
    margin = margin_width(reasons)
    shown = false
    for (symbol, range) in sort!(collect(ptx_functions(text)); by=first ∘ last)
        symbol == f.symbol || any(line -> haskey(counts, line), range) || continue
        shown && println(io)
        shown = true
        stack = SourceFrame[]
        for line in range
            code = text[line]
            # skip blanked directives and the labels that debug information refers to
            if !raw && (isempty(strip(code)) ||
                        occursin(r"^\$L__(tmp|func_begin|func_end)\d+:", code))
                continue
            end
            if frames && haskey(chains, line)
                print_frames(io, stack, chains[line], margin, "// ", kernel)
                stack = chains[line]
            end
            columns = if haskey(counts, line)
                samples, stalled = counts[line]
                hot = is_hot(sum(samples), total)
                sample_columns(reasons, samples, stalled, hot)
            else
                hot = false
                Line([(" "^margin, "")])
            end
            push!(columns, (code, hot ? bold_style : ""))
            print_line(io, columns, typemax(Int))
        end
    end
    println(io)
end

# the lines of every function defined in PTX code, by symbol
function ptx_functions(text::Vector{String})
    functions = Dict{String,UnitRange{Int}}()
    start = 0
    symbol = ""
    depth = 0       # of braces, which also delimit blocks in function bodies
    for (i, line) in enumerate(text)
        if start == 0
            m = match(r"^\s*(?:\.visible\s+|\.weak\s+)*\.(?:entry|func)\s+(?:\(.*?\)\s*)?([\w$]+)",
                      line)
            m === nothing && continue
            start = i
            symbol = m[1]
        end
        code = first(split(line, "//"; limit=2))
        if depth == 0 && occursin(';', code) && !occursin('{', code)
            start = 0   # a declaration
            continue
        end
        opened = depth > 0 || occursin('{', code)
        depth += count(==('{'), code) - count(==('}'), code)
        if opened && depth <= 0
            functions[symbol] = start:i
            start = 0
            depth = 0
        end
    end
    functions
end


## JSON

function Base.show(io::IO, ::MIME"application/json", p::PCProfile)
    buf = IOBuffer()
    json(buf, json_report(p))
    write(io, take!(buf))
    return
end

json_report(p::PCProfile) = barrier(json_pc_profile, p)
function json_pc_profile(p::PCProfile)
    frames = Any[]
    frame_ids = Dict{SourceFrame,Int}()
    frame_id(frame) = frame === unknown_frame ? nothing : get!(frame_ids, frame) do
        push!(frames, JSONObject("function" => frame.func, "file" => frame.file,
                                 "line" => frame.line))
        length(frames) - 1
    end
    chain_id(chain) = Any[frame_id(frame) for frame in chain]
    # the functions of a module share its PTX code
    ptx = Any[]
    ptx_ids = IdDict{Vector{String},Int}()
    ptx_id(code) = code === nothing ? nothing : get!(ptx_ids, code.text) do
        push!(ptx, join(code.text, '\n'))
        length(ptx) - 1
    end
    ptx_line(f, i) = f.ptx === nothing || f.ptx.lines[i] == 0 ? nothing : f.ptx.lines[i]
    counts(row) = JSONObject(Pair{String,Any}[p.reasons[j] => row[j]
                                              for j in eachindex(row) if row[j] != 0])

    functions = Any[]
    for f in p.functions
        r = report(f, p.format, p.noisefloor)
        r.count == 0 && continue
        locations = Any[JSONObject("depth" => r.rows.depth[i],
                                   "frame" => frame_id(r.rows.path[i][end]),
                                   "count" => r.rows.count[i],
                                   "stalled" => r.rows.stalled[i],
                                   "overhead" => r.rows.overhead[i],
                                   "reasons" => counts(r.rows.reasons[i]))
                        for i in eachindex(r.rows.depth)]
        instructions = Any[JSONObject("offset" => "0x" * string(f.instructions.offset[i]; base=16, pad=4),
                                      "sass" => f.instructions.sass[i],
                                      "frames" => chain_id(f.instructions.source[i]),
                                      "ptx_line" => ptx_line(f, i),
                                      "count" => counts(f.samples[i, :]),
                                      "stalled" => counts(f.stalled[i, :]))
                           for i in axes(f.samples, 1) if any(!iszero, @view f.samples[i, :])]
        push!(functions, JSONObject("name" => f.name, "symbol" => f.symbol,
                                    "count" => r.count, "stalled" => r.stalled,
                                    "ptx" => ptx_id(f.ptx), "locations" => locations,
                                    "instructions" => instructions))
    end

    JSONObject("schema_version" => 1, "view" => string(p.format), "device" => p.device,
               "period" => p.period, "noisefloor" => p.noisefloor,
               "total_samples" => p.total, "dropped_samples" => p.dropped,
               "reasons" => Any[p.reasons...], "frames" => frames, "ptx" => ptx,
               "functions" => functions)
end

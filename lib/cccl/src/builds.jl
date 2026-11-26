# building (JIT-compiling) algorithm instantiations
#
# Every combination of algorithm, value types and operators is compiled by HostJIT into a
# host library that launches CUB's kernels, which takes seconds. Builds only depend on the
# types involved, not on the data, so we cache them for the rest of the session.

const builds = Dict{Any,Any}()
const builds_lock = ReentrantLock()

# how long each build took, for diagnostics
const build_times = Pair{Any,Float64}[]

# whether to cache precompiled CUB headers on disk, which speeds up later builds
const use_pch = Ref(true)

function build_config(f)
    pch_dir = use_pch[] ? @get_scratch!("pch") : ""
    config = Ref(cccl_build_config(C_NULL, 0, C_NULL, 0, use_pch[], false,
                                   reinterpret(Cstring, pointer(pch_dir))))
    GC.@preserve pch_dir f(config)
end

# HostJIT writes diagnostics to the process' stderr, while its C API only returns an error
# code: on failure the compiler log, on success remarks about CUB's code that the build
# config cannot silence (it ignores compile flags other than `-D`). Capture them, so that a
# failure can report the log and a successful build stays quiet.
const hostjit_remark = r"^(\d+ (warning|error)s? generated when compiling for |warning: <unknown>:0:0: )"

function capture_diagnostics(f)
    mktemp() do path, io
        Libc.flush_cstdio()
        flush(stderr)
        saved = Libc.dup(RawFD(2))
        Libc.dup(fd(io), RawFD(2))
        failed = try
            f()
            false
        catch err
            err isa CuError || rethrow()
            true
        finally
            Libc.flush_cstdio()
            Libc.dup(saved, RawFD(2))
            ccall(:close, Cint, (RawFD,), saved)
        end

        output = read(path, String)
        failed && error("CCCL failed to build an algorithm:\n", strip(output))

        # pass on what others wrote to stderr in the meantime
        lines = split(output, '\n'; keepempty=false)
        remarks = filter(contains(hostjit_remark), lines)
        isempty(remarks) || @debug "HostJIT diagnostics:\n$(join(remarks, '\n'))"
        for line in lines
            contains(line, hostjit_remark) || println(stderr, line)
        end
    end
end

"""
    cached_build(builder, key, R)

Look up the build of type `R` for `key` (on the current device), or create it by calling
`builder(result::Ref{R}, cc_major, cc_minor, include_path, ctk_path, config)`.
"""
function cached_build(builder, key, ::Type{R}) where {R}
    dev = device()
    Base.@lock builds_lock begin
        get!(builds, (key, dev)) do
            cap = capability(dev)
            result = Ref{R}()
            include = "-I" * cccl_include_path()
            ctk = "-I" * joinpath(toolkit_path(), "include")
            t = @elapsed capture_diagnostics() do
                build_config() do config
                    builder(result, cap.major, cap.minor, include, ctk, config)
                end
            end
            push!(build_times, key => t)
            result[]
        end::R
    end
end

# call a CUB-style function twice: first to query the temporary storage it needs, then
# with that storage allocated
function with_temp_storage(f)
    nbytes = Ref{Csize_t}(0)
    f(C_NULL, nbytes)
    temp = CuArray{UInt8}(undef, max(nbytes[], 1))
    try
        f(reinterpret(Ptr{Cvoid}, pointer(temp)), nbytes)
    finally
        CUDACore.unsafe_free!(temp)
    end
    return
end

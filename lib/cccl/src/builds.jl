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

# the compiler's warnings are about CUB's code, and not actionable for users
const compile_flags = ["-w"]

function build_config(f)
    pch_dir = use_pch[] ? @get_scratch!("pch") : ""
    flags = map(flag -> reinterpret(Cstring, pointer(flag)), compile_flags)
    config = Ref(cccl_build_config(pointer(flags), length(flags), C_NULL, 0, use_pch[],
                                   false, reinterpret(Cstring, pointer(pch_dir))))
    GC.@preserve pch_dir flags f(config)
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
            t = @elapsed build_config() do config
                builder(result, cap.major, cap.minor, include, ctk, config)
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

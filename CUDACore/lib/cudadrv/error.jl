# Error type and decoding functionality

export CuError
@public description#, name


# an optional struct, used to represent e.g. optional error logs.
# this is to make CuErrors with/without additional logs compare equal
# (so that we can simply reuse `@test_throws CuError(code)`).
struct Optional{T}
    data::Union{Nothing,T}
    Optional{T}(data::Union{Nothing,T}=nothing) where {T} = new{T}(data)
end
Base.getindex(s::Optional) = s.data
function Base.isequal(a::Optional, b::Optional)
    if a.data === nothing || b.data === nothing
        return true
    else
        return isequal(a.data, b.data)
    end
end
Base.convert(::Type{Optional{T}}, ::Nothing) where T = Optional{T}()
Base.convert(::Type{Optional{T}}, x) where T = Optional{T}(convert(T, x))
Base.convert(::Type{Optional{T}}, x::Optional) where T = convert(Optional{T}, x[])


"""
    CuError(code)

Create a CUDA error object with error code `code`. Internally, errors thrown by API calls
are annotated with the driver's explanation of the failure, if the driver provides one.
"""
struct CuError <: Exception
    code::CUresult
    log::Optional{String}

    CuError(code, log=nothing) = new(code, log)
end

Base.convert(::Type{CUresult}, err::CuError) = err.code

Base.:(==)(x::CuError, y::CuError) = x.code == y.code
Base.hash(err::CuError, h::UInt) = hash(err.code, h)

"""
    name(err::CuError)

Gets the string representation of an error code.

```jldoctest
julia> err = CuError(CUDACore.cudaError_enum(1))
CuError(CUDA_ERROR_INVALID_VALUE)

julia> name(err)
"ERROR_INVALID_VALUE"
```
"""
function name(err::CuError)
    str_ref = Ref{Cstring}()
    cuGetErrorName(err, str_ref)
    unsafe_string(str_ref[])[6:end]
end

"""
    description(err::CuError)

Gets the string description of an error code.
"""
function description(err::CuError)
    if err.code == -1%UInt32
        "Cannot use the CUDA stub libraries."
    else
        str_ref = Ref{Cstring}()
        cuGetErrorString(err, str_ref)
        unsafe_string(str_ref[])
    end
end

function Base.showerror(io::IO, err::CuError)
    if !functional()
        # we might throw before the library is initialized
        print(io, "CUDA error (code $(reinterpret(Int32, err.code)), $(err.code))")
    else
        print(io, "CUDA error: $(description(err)) (code $(reinterpret(Int32, err.code)), $(name(err)))")
    end
    if err.log[] !== nothing
        print(io, "\nDriver log:")
        for line in eachline(IOBuffer(err.log[]))
            print(io, "\n  ", line)
        end
    end
end

Base.show(io::IO, ::MIME"text/plain", err::CuError) = print(io, "CuError($(err.code))")

@enum_without_prefix visibility=:public cudaError_enum CUDA_


## driver error log

# CUDA 12.9 introduced an error log: a ring buffer of up to 100 entries in which the driver
# explains, in plain English, why API calls failed. We dump it incrementally whenever we
# throw a CuError, so that the error message can include the driver's explanation. The log
# is process-wide, so messages from concurrent failures may be grouped in a single CuError.
#
# The same log can be written to a file by setting the `CUDA_LOG_FILE` environment variable.

const driver_log_lock = ReentrantLock()
const driver_log_iterator = Ref{CUlogIterator}(0)
const driver_log_initialized = Ref(false)

# return the entries the driver added to its log since the previous call (or since start-up,
# for the first call), or `nothing` if there are none or if the driver does not support this.
#
# NOTE: this consumes the log, so it should only be called when throwing a CuError, or to
#       discard entries from failures that were handled (see `discard_driver_log`).
function driver_log()
    # the log API was introduced in CUDA 12.9
    if !isassigned(_driver_version) || driver_version() < v"12.9"
        return nothing
    end

    # this function is called while throwing errors, so avoid our checked wrappers
    # (which might initialize state, or throw again).
    Base.@lock driver_log_lock begin
        buf = Vector{UInt8}(undef, 25_600)   # documented maximum
        sz = Ref{Csize_t}(length(buf))
        if driver_log_initialized[]
            res = @gcsafe_ccall libcuda.cuLogsDumpToMemory(
                driver_log_iterator::Ptr{CUlogIterator}, buf::Ptr{UInt8}, sz::Ptr{Csize_t},
                0::Cuint)::CUresult
        else
            # first call: dump everything, and remember where the log currently ends
            res = @gcsafe_ccall libcuda.cuLogsDumpToMemory(C_NULL::Ptr{CUlogIterator},
                                                           buf::Ptr{UInt8}, sz::Ptr{Csize_t},
                                                           0::Cuint)::CUresult
            res == SUCCESS || return nothing
            res = @gcsafe_ccall libcuda.cuLogsCurrent(
                driver_log_iterator::Ptr{CUlogIterator}, 0::Cuint)::CUresult
            if res == SUCCESS
                driver_log_initialized[] = true
            end
        end
        res == SUCCESS || return nothing

        # the reported size excludes the NUL terminator; tolerate drivers that include it
        n = Int(sz[])
        while n > 0 && buf[n] == 0
            n -= 1
        end
        log = rstrip(String(resize!(buf, n)))
        return isempty(log) ? nothing : log
    end
end

# discard log entries from API failures that were handled, so that they do not get
# attributed to the next error that is thrown.
discard_driver_log() = (driver_log(); nothing)

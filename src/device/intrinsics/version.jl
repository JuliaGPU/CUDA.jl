# device intrinsics for querying the compute SimpleVersion and PTX ISA version


## a GPU-compatible version number

export SimpleVersion, @sv_str

struct SimpleVersion
    major::UInt32
    minor::UInt32

    SimpleVersion(major, minor=0) = new(major, minor)
end

function Base.tryparse(::Type{SimpleVersion}, v::AbstractString)
    parts = split(v, ".")
    1 <= length(parts) <= 2 || return nothing

    int_parts = map(parts) do part
        tryparse(Int, part)
    end
    any(isnothing, int_parts) && return nothing

    SimpleVersion(int_parts...)
end

function Base.parse(::Type{SimpleVersion}, v::AbstractString)
    ver = tryparse(SimpleVersion, v)
    ver === nothing && throw(ArgumentError("invalid SimpleVersion string: '$v'"))
    return ver
end

SimpleVersion(v::AbstractString) = parse(SimpleVersion, v)

@inline function Base.isless(a::SimpleVersion, b::SimpleVersion)
    (a.major < b.major) && return true
    (a.major > b.major) && return false
    (a.minor < b.minor) && return true
    (a.minor > b.minor) && return false
    return false
end

macro sv_str(str)
    SimpleVersion(str)
end


## accessors for the compute SimpleVersion and PTX ISA version

export compute_capability, ptx_isa_version

for var in ["sm_major", "sm_minor", "ptx_major", "ptx_minor"]
    @eval @inline $(Symbol(var))() =
        Base.llvmcall(
            $("""@$var = external global i32
                 define i32 @entry() #0 {
                     %val = load i32, i32* @$var
                     ret i32 %val
                 }
                 attributes #0 = { alwaysinline }
            """, "entry"), UInt32, Tuple{})
end

@device_function @inline compute_capability() = SimpleVersion(sm_major(), sm_minor())
@device_function @inline ptx_isa_version() = SimpleVersion(ptx_major(), ptx_minor())


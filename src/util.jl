## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

@static if VERSION >= v"0.7.0-DEV.4003"
    import Base.(:)
    (:)(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)
else
    import Base.colon
    colon(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)
end

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)

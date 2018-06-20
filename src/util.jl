## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

import Base.(:)
(:)(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)

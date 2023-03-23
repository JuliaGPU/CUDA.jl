export struct_size

Base.@pure function find_field(::Type{T}, fieldname) where T
    findfirst(f->f === fieldname, fieldnames(T))
end

function struct_size(type, lastfield)
    field = find_field(type, lastfield)
    @assert field !== nothing
    fieldoffset(type, field) + sizeof(fieldtype(type, field))
end
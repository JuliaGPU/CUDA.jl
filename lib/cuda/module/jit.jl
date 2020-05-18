@enum_without_prefix CUjit_option CU_
@enum_without_prefix CUjitInputType CU_

function convert_bits(::Type{T}, data::UInt) where T
    if sizeof(data) == sizeof(T)
        return reinterpret(T, data)
    elseif sizeof(data) == 8 && sizeof(T) == 4
        relevant = UInt32(data & 0x00000000ffffffff)
        return reinterpret(T, relevant)
    else
        error("don't know how to extract $(sizeof(T)) bytes out of $(sizeof(data))")
    end
end

function encode(options::Dict{CUjit_option,Any})
    keys = Vector{CUjit_option}()
    vals = Vector{Ptr{Cvoid}}()

    for (opt, val) in options
        push!(keys, opt)
        if opt == JIT_GENERATE_LINE_INFO ||
           opt == JIT_GENERATE_DEBUG_INFO ||
           opt == JIT_LOG_VERBOSE
            push!(vals, convert(Ptr{Cvoid}, convert(Int, val::Bool)))
        elseif opt == JIT_INFO_LOG_BUFFER
            buf = val::Vector{UInt8}
            push!(vals, pointer(buf))
            push!(keys, JIT_INFO_LOG_BUFFER_SIZE_BYTES)
            push!(vals, sizeof(buf))
        elseif opt == JIT_ERROR_LOG_BUFFER
            buf = val::Vector{UInt8}
            push!(vals, pointer(buf))
            push!(keys, JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
            push!(vals, sizeof(buf))
        elseif isa(val, Ptr{Cvoid})
            push!(vals, val)
        else
            error("cannot handle option $opt")
        end
    end

    @assert length(keys) == length(vals)
    return keys, vals
end

function decode(keys::Vector{CUjit_option}, vals::Vector{Ptr{Cvoid}})
    @assert length(keys) == length(vals)
    options = Dict{CUjit_option,Any}()

    # decode the raw option value bits to their proper type
    for (opt, val) = zip(keys, vals)
        data = reinterpret(UInt, val)
        if opt == JIT_WALL_TIME
            options[opt] = convert_bits(Cfloat, data)
        elseif opt == JIT_GENERATE_LINE_INFO ||
           opt == JIT_GENERATE_DEBUG_INFO ||
           opt == JIT_LOG_VERBOSE
            options[opt] = Bool(convert_bits(Cint, data))
        elseif opt == JIT_INFO_LOG_BUFFER_SIZE_BYTES ||
           opt == JIT_ERROR_LOG_BUFFER_SIZE_BYTES
            options[opt] = convert_bits(Cuint, data)
        elseif opt == JIT_INFO_LOG_BUFFER ||
           opt == JIT_ERROR_LOG_BUFFER
            options[opt] = reinterpret(Ptr{UInt8}, data)
        else
            options[opt] = data
        end
    end

    # convert some values to easier-to-handle types
    if haskey(options, JIT_INFO_LOG_BUFFER)
        buf = options[JIT_INFO_LOG_BUFFER]
        size = options[JIT_INFO_LOG_BUFFER_SIZE_BYTES]
        delete!(options, JIT_INFO_LOG_BUFFER_SIZE_BYTES)
        options[JIT_INFO_LOG_BUFFER] = unsafe_string(buf, Int(size / sizeof(UInt8)))
    end
    if haskey(options, JIT_ERROR_LOG_BUFFER)
        buf = options[JIT_ERROR_LOG_BUFFER]
        size = options[JIT_ERROR_LOG_BUFFER_SIZE_BYTES]
        delete!(options, JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        options[JIT_ERROR_LOG_BUFFER] = unsafe_string(buf, Int(size / sizeof(UInt8)))
    end

    return options
end

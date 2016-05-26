@enum(CUjit_option, CU_JIT_MAX_REGISTERS = Cint(0),
                    CU_JIT_THREADS_PER_BLOCK,
                    CU_JIT_WALL_TIME,
                    CU_JIT_INFO_LOG_BUFFER,
                    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
                    CU_JIT_ERROR_LOG_BUFFER,
                    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                    CU_JIT_OPTIMIZATION_LEVEL,
                    CU_JIT_TARGET_FROM_CUCONTEXT,
                    CU_JIT_TARGET,
                    CU_JIT_FALLBACK_STRATEGY,
                    CU_JIT_GENERATE_DEBUG_INFO,
                    CU_JIT_LOG_VERBOSE,
                    CU_JIT_GENERATE_LINE_INFO,
                    CU_JIT_CACHE_MODE)

function convert_bits{T}(::Type{T}, data::UInt)
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
    keys = Array(CUjit_option, 0)
    vals = Array(Ptr{Void}, 0)

    for (opt, val) in options
        push!(keys, opt)
        if opt == CU_JIT_GENERATE_LINE_INFO ||
           opt == CU_JIT_GENERATE_DEBUG_INFO ||
           opt == CU_JIT_LOG_VERBOSE
            push!(vals, convert(Ptr{Void}, convert(Int, val::Bool)))
        elseif opt == CU_JIT_INFO_LOG_BUFFER
            buf = val::Vector{UInt8}
            push!(vals, pointer(buf))
            push!(keys, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
            println(sizeof(buf))
            println(convert(Ptr{Void}, sizeof(buf)))
            push!(vals, sizeof(buf))
        elseif opt == CU_JIT_ERROR_LOG_BUFFER
            buf = val::Vector{UInt8}
            push!(vals, pointer(buf))
            push!(keys, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
            push!(vals, sizeof(buf))
        elseif isa(val, Ptr{Void})
            push!(vals, val)
        else
            error("cannot handle option $opt")
        end
    end

    @assert length(keys) == length(vals)
    return keys, vals
end

function decode(keys::Array{CUjit_option,1}, vals::Array{Ptr{Void}, 1})
    @assert length(keys) == length(vals)
    options = Dict{CUjit_option,Any}()

    # decode the raw option value bits to their proper type
    for (opt, val) = zip(keys, vals)
        data = reinterpret(UInt, val)
        if opt == CU_JIT_WALL_TIME
            options[opt] = convert_bits(Cfloat, data)
        elseif opt == CU_JIT_GENERATE_LINE_INFO ||
           opt == CU_JIT_GENERATE_DEBUG_INFO ||
           opt == CU_JIT_LOG_VERBOSE
            options[opt] = Bool(convert_bits(Cint, data))
        elseif opt == CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES ||
           opt == CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES
            options[opt] = convert_bits(Cuint, data)
        elseif opt == CU_JIT_INFO_LOG_BUFFER ||
           opt == CU_JIT_ERROR_LOG_BUFFER
            options[opt] = reinterpret(Ptr{UInt8}, data)
        else
            options[opt] = data
        end
    end

    # convert some values to easier-to-handle types 
    if haskey(options, CU_JIT_INFO_LOG_BUFFER)
        buf = options[CU_JIT_INFO_LOG_BUFFER]
        size = options[CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES]
        delete!(options, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES)
        options[CU_JIT_INFO_LOG_BUFFER] = String(buf, Int(size / sizeof(UInt8)))
    end
    if haskey(options, CU_JIT_ERROR_LOG_BUFFER)
        buf = options[CU_JIT_ERROR_LOG_BUFFER]
        size = options[CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES]
        delete!(options, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES)
        options[CU_JIT_ERROR_LOG_BUFFER] = String(buf, Int(size / sizeof(UInt8)))
    end

    return options
end

export CUDAError

struct CUDAError <: Exception
    code::CUresult
    msg::AbstractString
end
Base.show(io::IO, err::CUDAError) = print(io, "CUDAError(code $(err.code), $(err.msg))")

function CUDAError(result::CUresult)
    msg = error_string(result)
    return CUDAError(result, msg)
end

function error_string(result)
    str_ref = Ref{Cstring}()
    cuGetErrorString(result, str_ref)
    unsafe_string(str_ref[])
end

macro check(ex)
    quote
        local err::CUresult
        res = $(esc(ex))
        if res != CUDA_SUCCESS
            throw(CUDAError(res))
        end
        res
    end
end

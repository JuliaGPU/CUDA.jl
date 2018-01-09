export CUDNNError

struct CUDNNError <: Exception
    code::cudnnStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUDNNError) = print(io, "CUDNNError(code $(err.code), $(err.msg))")

function CUDNNError(status::cudnnStatus_t)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUDNNError(status, msg)
end

macro check(dnn_func)
    quote
        local err::cudnnStatus_t
        err = $(esc(dnn_func))
        if err != CUDNN_STATUS_SUCCESS
            throw(CUDNNError(err))
        end
        err
    end
end

export CUSOLVERError

struct CUSOLVERError <: Exception
    code::cudnnStatus_t
    msg::AbstractString
end
Base.show(io::IO, err::CUSOLVERError) = print(io, "CUSOLVERError(code $(err.code), $(err.msg))")

function CUSOLVERError(code::cudnnStatus_t)
    msg = unsafe_string(cudnnGetErrorString(status))
    return CUSOLVERError(code, msg)
end

macro check(dnn_func)
    quote
        local err::cudnnStatus_t
        err = $(esc(dnn_func::Expr))
        if err != CUDNN_STATUS_SUCCESS
            throw(CUSOLVERError(err))
        end
        err
    end
end
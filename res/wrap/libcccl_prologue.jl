# the C library reports failures as driver API errors, with details printed to stderr
@inline function check(f)
    res = f()
    if res != CUDA_SUCCESS
        throw_api_error(res)
    end
    return
end

@noinline throw_api_error(res) = throw(CuError(res))

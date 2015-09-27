
typealias curandStatus_t UInt32


function statuscheck(status::UInt32)
    if status != CURAND_STATUS_SUCCESS        
        # Because try/finally may disguise the source of the problem,
        # let's show a backtrace here
        warn("CURAND error triggered from:")
        Base.show_backtrace(STDOUT, backtrace())
        println("(status was $status)")
        throw("It's bad!")
        # TODO
        # throw(statusmessage(status))
    end    
end
    

type RNG
    ptr::Ptr{Void}
    rng_type::Int
end

function create_generator(rng_type=CURAND_RNG_PSEUDO_DEFAULT)
    aptr = Ptr{Void}[0]
    statuscheck(ccall((:curandCreateGenerator, libcurand),
                      curandStatus_t, (Ptr{Void}, Cint), aptr, rng_type))
    return RNG(aptr[1], rng_type)
end

# TODO: curandCreateGeneratorHost

function destroy_generator(rng::RNG)    
    statuscheck(ccall((:curandDestroyGenerator, libcurand),
                      curandStatus_t, (Ptr{Void},), rng.ptr))
end


function get_version()
    ver = Ref{Cint}(0)
    statuscheck(ccall((:curandGetVersion, libcurand),
                      curandStatus_t, (Ref{Cint},), ver))
    return ver[]
end

# TODO: curandSetStream

function set_pseudo_random_generator_seed(rng::RNG, seed::Int64)
    statuscheck(ccall((:curandSetPseudoRandomGeneratorSeed, libcurand),
                      curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, seed))
end

function set_generator_offset(rng::RNG, offset::Int64)
    statuscheck(ccall((:curandSetGeneratorOffset, libcurand),
                      curandStatus_t, (Ptr{Void}, Clonglong), rng.ptr, offset))
end

# TODO: curandSetGeneratorOrdering
# TODO: curandSetQuasiRandomGeneratorDimensions

"""
Generate 64-bit quasirandom numbers.
"""
function generate(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CudaArray(UInt32, sz)
    statuscheck(ccall((:curandGenerate, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{UInt32}, Csize_t),
                      rng.ptr, arr.ptr.ptr, n))
    return arr
end

generate(rng::RNG, n::Int) = generate(rng, UInt(n))


"""
Generate uniformly distributed floats.
"""
function generate_long_long(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CudaArray(UInt64, sz)
    # TODO: results in type error (on CUDA side)
    statuscheck(ccall((:curandGenerateLongLong, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{UInt64}, Csize_t),
                      rng.ptr, arr.ptr.ptr, n))
    return arr
end

generate_long_long(rng::RNG, n::Int) = generate_long_long(rng, UInt(n))



"""
Generate uniformly distributed floats. 
"""
function generate_uniform(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CudaArray(Float32, sz)    
    statuscheck(ccall((:curandGenerateUniform, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{Float32}, Csize_t),
                      rng.ptr, arr.ptr.ptr, n))
    return arr
end

"""
Generate uniformly distributed doubles. 
"""
function generate_uniform_double(rng::RNG, n::UInt)
    sz = Int(n)
    arr = CudaArray(Float64, sz)    
    statuscheck(ccall((:curandGenerateUniformDouble, libcurand),
                      curandStatus_t, (Ptr{Void}, Ptr{Float64}, Csize_t),
                      rng.ptr, arr.ptr.ptr, n))
    return arr
end


"""
Generate normally distributed floats. 
"""
function generate_normal(rng::RNG, n::UInt, mean::Float32, stddev::Float32)
    sz = Int(n)
    arr = CudaArray(Float32, sz)    
    statuscheck(ccall((:curandGenerateNormal, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float32}, Csize_t, Cfloat, Cfloat),
                      rng.ptr, arr.ptr.ptr, n, mean, stddev))
    return arr
end

"""
Generate normally distributed doubles. 
"""
function generate_normal_double(rng::RNG, n::UInt, mean::Float64, stddev::Float64)
    sz = Int(n)
    arr = CudaArray(Float64, sz)    
    statuscheck(ccall((:curandGenerateNormalDouble, libcurand),
                      curandStatus_t,
                      (Ptr{Void}, Ptr{Float64}, Csize_t, Cdouble, Cdouble),
                      rng.ptr, arr.ptr.ptr, n, mean, stddev))
    return arr
end



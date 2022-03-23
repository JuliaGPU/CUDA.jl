
@checked function cuFileDriverSetMaxDirectIOSize(max_direct_io_size)
    initialize_context()
    ccall((:cuFileDriverSetMaxDirectIOSize, libcufile), CUfileError_t, (Csize_t,), max_direct_io_size)
end

@checked function cuFileDriverSetMaxCacheSize(max_cache_size)
    initialize_context()
    ccall((:cuFileDriverSetMaxCacheSize, libcufile), CUfileError_t, (Csize_t,), max_cache_size)
end

@checked function cuFileDriverGetProperties(props)
    initialize_context()
    ccall((:cuFileDriverGetProperties, libcufile), CUfileError_t, (Ptr{CUfileDrvProps_t},), props)
end

@checked function cuFileBufDeregister(devPtr_base)
    initialize_context()
    ccall((:cuFileBufDeregister, libcufile), CUfileError_t, (Ptr{Cvoid},), devPtr_base)
end

function cuFileRead(fh, devPtr_base, size, file_offset, devPtr_offset)
    initialize_context()
    ccall((:cuFileRead, libcufile), Cssize_t, (CUfileHandle_t, Ptr{Cvoid}, Csize_t, off_t, off_t), fh, devPtr_base, size, file_offset, devPtr_offset)
end

@checked function cuFileDriverSetPollMode(poll, poll_threshold_size)
    initialize_context()
    ccall((:cuFileDriverSetPollMode, libcufile), CUfileError_t, (Bool, Csize_t), poll, poll_threshold_size)
end

@checked function cuFileBufRegister(devPtr_base, length, flags)
    initialize_context()
    ccall((:cuFileBufRegister, libcufile), CUfileError_t, (Ptr{Cvoid}, Csize_t, Cint), devPtr_base, length, flags)
end

@checked function cuFileDriverClose()
    initialize_context()
    ccall((:cuFileDriverClose, libcufile), CUfileError_t, ())
end

@checked function cuFileDriverSetMaxPinnedMemSize(max_pinned_size)
    initialize_context()
    ccall((:cuFileDriverSetMaxPinnedMemSize, libcufile), CUfileError_t, (Csize_t,), max_pinned_size)
end

function cuFileWrite(fh, devPtr_base, size, file_offset, devPtr_offset)
    initialize_context()
    ccall((:cuFileWrite, libcufile), Cssize_t, (CUfileHandle_t, Ptr{Cvoid}, Csize_t, off_t, off_t), fh, devPtr_base, size, file_offset, devPtr_offset)
end

@checked function cuFileHandleRegister(fh, descr)
    initialize_context()
    ccall((:cuFileHandleRegister, libcufile), CUfileError_t, (Ptr{CUfileHandle_t}, Ptr{CUfileDescr_t}), fh, descr)
end

@checked function cuFileDriverOpen()
    initialize_context()
    ccall((:cuFileDriverOpen, libcufile), CUfileError_t, ())
end

function cufileop_status_error(status)
    initialize_context()
    ccall((:cufileop_status_error, libcufile), Cstring, (CUfileOpError,), status)
end

function cuFileHandleDeregister(fh)
    initialize_context()
    ccall((:cuFileHandleDeregister, libcufile), Cvoid, (CUfileHandle_t,), fh)
end

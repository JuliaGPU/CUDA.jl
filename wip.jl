using CUDA
using .CUFILE

using Base: Filesystem, JL_O_CREAT, JL_O_WRONLY

const JL_O_DIRECT = 0x0004000   # XXX: add to file_constants.h?

function main()
    path = "/mnt/gds/testfile"  # tmpfs doesn't support O_DIRECT
    file = Filesystem.open(path, JL_O_CREAT|JL_O_WRONLY|JL_O_DIRECT, 0o644)
    file_fd = reinterpret(Cint, fd(file))

    CUFILE.cuFileDriverOpen()

    desc = Ref(CUFILE.CUfileDescr_t(CUFILE.CU_FILE_HANDLE_TYPE_OPAQUE_FD,
                                    CUFILE.CUfileDescrHandle_t(Int(file_fd)), C_NULL))
    handle = Ref{CUFILE.CUfileHandle_t}()
    CUFILE.cuFileHandleRegister(handle, desc)

    return
end

isinteractive() || main()

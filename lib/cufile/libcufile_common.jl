# Automatically generated using Clang.jl

const CUFILEOP_BASE_ERR = 5000

# Skipping MacroDefinition: CUFILEOP_STATUS_ENTRIES CUFILE_OP ( 0 , CU_FILE_SUCCESS , cufile success ) CUFILE_OP ( CUFILEOP_BASE_ERR + 1 , CU_FILE_DRIVER_NOT_INITIALIZED , nvidia - fs driver is not loaded ) CUFILE_OP ( CUFILEOP_BASE_ERR + 2 , CU_FILE_DRIVER_INVALID_PROPS , invalid property ) CUFILE_OP ( CUFILEOP_BASE_ERR + 3 , CU_FILE_DRIVER_UNSUPPORTED_LIMIT , property range error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 4 , CU_FILE_DRIVER_VERSION_MISMATCH , nvidia - fs driver version mismatch ) CUFILE_OP ( CUFILEOP_BASE_ERR + 5 , CU_FILE_DRIVER_VERSION_READ_ERROR , nvidia - fs driver version read error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 6 , CU_FILE_DRIVER_CLOSING , driver shutdown in progress ) CUFILE_OP ( CUFILEOP_BASE_ERR + 7 , CU_FILE_PLATFORM_NOT_SUPPORTED , GPUDirect Storage not supported on current platform ) CUFILE_OP ( CUFILEOP_BASE_ERR + 8 , CU_FILE_IO_NOT_SUPPORTED , GPUDirect Storage not supported on current file ) CUFILE_OP ( CUFILEOP_BASE_ERR + 9 , CU_FILE_DEVICE_NOT_SUPPORTED , GPUDirect Storage not supported on current GPU ) CUFILE_OP ( CUFILEOP_BASE_ERR + 10 , CU_FILE_NVFS_DRIVER_ERROR , nvidia - fs driver ioctl error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 11 , CU_FILE_CUDA_DRIVER_ERROR , CUDA Driver API error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 12 , CU_FILE_CUDA_POINTER_INVALID , invalid device pointer ) CUFILE_OP ( CUFILEOP_BASE_ERR + 13 , CU_FILE_CUDA_MEMORY_TYPE_INVALID , invalid pointer memory type ) CUFILE_OP ( CUFILEOP_BASE_ERR + 14 , CU_FILE_CUDA_POINTER_RANGE_ERROR , pointer range exceeds allocated address range ) CUFILE_OP ( CUFILEOP_BASE_ERR + 15 , CU_FILE_CUDA_CONTEXT_MISMATCH , cuda context mismatch ) CUFILE_OP ( CUFILEOP_BASE_ERR + 16 , CU_FILE_INVALID_MAPPING_SIZE , access beyond maximum pinned size ) CUFILE_OP ( CUFILEOP_BASE_ERR + 17 , CU_FILE_INVALID_MAPPING_RANGE , access beyond mapped size ) CUFILE_OP ( CUFILEOP_BASE_ERR + 18 , CU_FILE_INVALID_FILE_TYPE , unsupported file type ) CUFILE_OP ( CUFILEOP_BASE_ERR + 19 , CU_FILE_INVALID_FILE_OPEN_FLAG , unsupported file open flags ) CUFILE_OP ( CUFILEOP_BASE_ERR + 20 , CU_FILE_DIO_NOT_SET , fd direct IO not set ) CUFILE_OP ( CUFILEOP_BASE_ERR + 22 , CU_FILE_INVALID_VALUE , invalid arguments ) CUFILE_OP ( CUFILEOP_BASE_ERR + 23 , CU_FILE_MEMORY_ALREADY_REGISTERED , device pointer already registered ) CUFILE_OP ( CUFILEOP_BASE_ERR + 24 , CU_FILE_MEMORY_NOT_REGISTERED , device pointer lookup failure ) CUFILE_OP ( CUFILEOP_BASE_ERR + 25 , CU_FILE_PERMISSION_DENIED , driver or file access error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 26 , CU_FILE_DRIVER_ALREADY_OPEN , driver is already open ) CUFILE_OP ( CUFILEOP_BASE_ERR + 27 , CU_FILE_HANDLE_NOT_REGISTERED , file descriptor is not registered ) CUFILE_OP ( CUFILEOP_BASE_ERR + 28 , CU_FILE_HANDLE_ALREADY_REGISTERED , file descriptor is already registered ) CUFILE_OP ( CUFILEOP_BASE_ERR + 29 , CU_FILE_DEVICE_NOT_FOUND , GPU device not found ) CUFILE_OP ( CUFILEOP_BASE_ERR + 30 , CU_FILE_INTERNAL_ERROR , internal error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 31 , CU_FILE_GETNEWFD_FAILED , failed to obtain new file descriptor ) CUFILE_OP ( CUFILEOP_BASE_ERR + 33 , CU_FILE_NVFS_SETUP_ERROR , NVFS driver initialization error ) CUFILE_OP ( CUFILEOP_BASE_ERR + 34 , CU_FILE_IO_DISABLED , GPUDirect Storage disabled by config on current file )
# Skipping MacroDefinition: CUFILE_OP ( code , name , string ) name = code ,
# Skipping MacroDefinition: CUFILE_OP ( code , name , string ) case name : return # string ;
# Skipping MacroDefinition: IS_CUFILE_ERR ( err ) ( abs ( ( err ) ) > CUFILEOP_BASE_ERR )
# Skipping MacroDefinition: CUFILE_ERRSTR ( err ) cufileop_status_error ( ( CUfileOpError ) abs ( ( err ) ) )
# Skipping MacroDefinition: IS_CUDA_ERR ( status ) ( ( status ) . err == CU_FILE_CUDA_DRIVER_ERROR )
# Skipping MacroDefinition: CU_FILE_CUDA_ERR ( status ) ( ( status ) . cu_err )

const CU_FILE_RDMA_REGISTER = 1
const CU_FILE_RDMA_RELAXED_ORDERING = 1 << 1

@cenum CUfileOpError::UInt32 begin
    CU_FILE_SUCCESS = 0
    CU_FILE_DRIVER_NOT_INITIALIZED = 5001
    CU_FILE_DRIVER_INVALID_PROPS = 5002
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = 5003
    CU_FILE_DRIVER_VERSION_MISMATCH = 5004
    CU_FILE_DRIVER_VERSION_READ_ERROR = 5005
    CU_FILE_DRIVER_CLOSING = 5006
    CU_FILE_PLATFORM_NOT_SUPPORTED = 5007
    CU_FILE_IO_NOT_SUPPORTED = 5008
    CU_FILE_DEVICE_NOT_SUPPORTED = 5009
    CU_FILE_NVFS_DRIVER_ERROR = 5010
    CU_FILE_CUDA_DRIVER_ERROR = 5011
    CU_FILE_CUDA_POINTER_INVALID = 5012
    CU_FILE_CUDA_MEMORY_TYPE_INVALID = 5013
    CU_FILE_CUDA_POINTER_RANGE_ERROR = 5014
    CU_FILE_CUDA_CONTEXT_MISMATCH = 5015
    CU_FILE_INVALID_MAPPING_SIZE = 5016
    CU_FILE_INVALID_MAPPING_RANGE = 5017
    CU_FILE_INVALID_FILE_TYPE = 5018
    CU_FILE_INVALID_FILE_OPEN_FLAG = 5019
    CU_FILE_DIO_NOT_SET = 5020
    CU_FILE_INVALID_VALUE = 5022
    CU_FILE_MEMORY_ALREADY_REGISTERED = 5023
    CU_FILE_MEMORY_NOT_REGISTERED = 5024
    CU_FILE_PERMISSION_DENIED = 5025
    CU_FILE_DRIVER_ALREADY_OPEN = 5026
    CU_FILE_HANDLE_NOT_REGISTERED = 5027
    CU_FILE_HANDLE_ALREADY_REGISTERED = 5028
    CU_FILE_DEVICE_NOT_FOUND = 5029
    CU_FILE_INTERNAL_ERROR = 5030
    CU_FILE_GETNEWFD_FAILED = 5031
    CU_FILE_NVFS_SETUP_ERROR = 5033
    CU_FILE_IO_DISABLED = 5034
end

struct CUfileError
    err::CUfileOpError
    cu_err::CUresult
end

const CUfileError_t = CUfileError

@cenum CUfileDriverStatusFlags::UInt32 begin
    CU_FILE_LUSTRE_SUPPORTED = 0
    CU_FILE_WEKAFS_SUPPORTED = 1
    CU_FILE_NFS_SUPPORTED = 2
    CU_FILE_GPFS_SUPPORTED = 3
    CU_FILE_NVME_SUPPORTED = 4
    CU_FILE_NVMEOF_SUPPORTED = 5
    CU_FILE_SCSI_SUPPORTED = 6
    CU_FILE_SCALEFLUX_CSD_SUPPORTED = 7
    CU_FILE_NVMESH_SUPPORTED = 8
end

const CUfileDriverStatusFlags_t = CUfileDriverStatusFlags

@cenum CUfileDriverControlFlags::UInt32 begin
    CU_FILE_USE_POLL_MODE = 0
    CU_FILE_ALLOW_COMPAT_MODE = 1
end

const CUfileDriverControlFlags_t = CUfileDriverControlFlags

@cenum CUfileFeatureFlags::UInt32 begin
    CU_FILE_DYN_ROUTING_SUPPORTED = 0
    CU_FILE_BATCH_IO_SUPPORTED = 1
    CU_FILE_STREAMS_SUPPORTED = 2
end

const CUfileFeatureFlags_t = CUfileFeatureFlags

struct CUfileDrvPropsNvfs
    major_version::UInt32
    minor_version::UInt32
    poll_thresh_size::Csize_t
    max_direct_io_size::Csize_t
    dstatusflags::UInt32
    dcontrolflags::UInt32
end

struct CUfileDrvProps
    nvfs::CUfileDrvPropsNvfs
    fflags::UInt32
    max_device_cache_size::UInt32
    per_buffer_cache_size::UInt32
    max_device_pinned_mem_size::UInt32
    max_batch_io_timeout_msecs::UInt32
end

const CUfileDrvProps_t = CUfileDrvProps

struct cufileRDMAInfo
    version::Cint
    desc_len::Cint
    desc_str::Cstring
end

const cufileRDMAInfo_t = cufileRDMAInfo

struct CUfileFSOps
    fs_type::Ptr{Cvoid}
    getRDMADeviceList::Ptr{Cvoid}
    getRDMADevicePriority::Ptr{Cvoid}
    read::Ptr{Cvoid}
    write::Ptr{Cvoid}
end

const CUfileFSOps_t = CUfileFSOps

@cenum CUfileFileHandleType::UInt32 begin
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_FILE_HANDLE_TYPE_USERSPACE_FS = 3
end

struct CUfileDescrHandle_t
    handle::Ptr{Cvoid}
end

struct CUfileDescr_t
    type::CUfileFileHandleType
    handle::CUfileDescrHandle_t
    fs_ops::Ptr{CUfileFSOps_t}
end

const CUfileHandle_t = Ptr{Cvoid}

const IS_LIBC_MUSL = occursin("musl", Base.MACHINE)

if Sys.islinux() && Sys.ARCH === :aarch64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :aarch64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && startswith(string(Sys.ARCH), "arm") && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.islinux() && Sys.ARCH === :i686 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :i686 && IS_LIBC_MUSL
    const off_t = Clonglong
elseif Sys.iswindows() && Sys.ARCH === :i686
    const off32_t = Clong
    const off_t = off32_t
elseif Sys.islinux() && Sys.ARCH === :powerpc64le
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.isapple()
    const __darwin_off_t = Int64
    const off_t = __darwin_off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && !IS_LIBC_MUSL
    const __off_t = Clong
    const off_t = __off_t
elseif Sys.islinux() && Sys.ARCH === :x86_64 && IS_LIBC_MUSL
    const off_t = Clong
elseif Sys.isbsd() && !Sys.isapple()
    const __off_t = Int64
    const off_t = __off_t
elseif Sys.iswindows() && Sys.ARCH === :x86_64
    const off32_t = Clong
    const off_t = off32_t
end

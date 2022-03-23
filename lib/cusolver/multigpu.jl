
## wrappers for multi-gpu functionality in cusolverMg


# auxiliary functionality

# NOTE: in the cublasMg preview, which also relies on this functionality, a separate library
#       called 'cudalibmg' is introduced. factor this out when we actually ship that.

mutable struct MatrixDescriptor
    desc::cudaLibMgMatrixDesc_t

    function MatrixDescriptor(a, grid; rowblocks = size(a, 1), colblocks = size(a, 2), elta=eltype(a) )
        desc = Ref{cudaLibMgMatrixDesc_t}()
        cusolverMgCreateMatrixDesc(desc, size(a, 1), size(a, 2), rowblocks, colblocks, elta, grid)
        return new(desc[])
    end
end

Base.unsafe_convert(::Type{cudaLibMgMatrixDesc_t}, obj::MatrixDescriptor) = obj.desc

mutable struct DeviceGrid
    desc::cudaLibMgGrid_t

    function DeviceGrid(num_row_devs, num_col_devs, deviceIds, mapping)
        @assert num_row_devs == 1 "Only 1-D column block cyclic is supported, so numRowDevices must be equal to 1."
        desc = Ref{cudaLibMgGrid_t}()
        cusolverMgCreateDeviceGrid(desc, num_row_devs, num_col_devs, deviceIds, mapping)
        return new(desc[])
    end
end

Base.unsafe_convert(::Type{cudaLibMgGrid_t}, obj::DeviceGrid) = obj.desc

function allocateBuffers(n_row_devs, n_col_devs, mat::Matrix)
    mat_row_block_size = div(size(mat, 1), n_row_devs)
    mat_col_block_size = div(size(mat, 2), n_col_devs)
    mat_buffers  = Vector{CuMatrix{eltype(mat)}}(undef, ndevices())
    mat_numRows  = Vector{Int64}(undef, ndevices())
    mat_numCols  = Vector{Int64}(undef, ndevices())
    typesize = sizeof(eltype(mat))
    ldas = Vector{Int64}(undef, ndevices())
    current_dev = device()
    mat_cpu_bufs = Vector{Matrix{eltype(mat)}}(undef, ndevices())
    for (di, dev) in enumerate(devices())
        ldas[di]    = mat_col_block_size
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1

        mat_row_inds     = ((dev_row-1)*mat_row_block_size+1):min(dev_row*mat_row_block_size, size(mat, 1))
        mat_col_inds     = ((dev_col-1)*mat_col_block_size+1):min(dev_col*mat_col_block_size, size(mat, 2))
        mat_cpu_bufs[di] = Array(mat[mat_row_inds, mat_col_inds])
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        mat_gpu_buf = CuMatrix{eltype(mat)}(undef, size(mat_cpu_bufs[di]))
        copyto!(mat_gpu_buf, mat_cpu_bufs[di])
        mat_buffers[di] = mat_gpu_buf
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        synchronize()
    end
    device!(current_dev)
    return mat_buffers
end

function returnBuffers(n_row_devs, n_col_devs, row_block_size, col_block_size, dDs, D)
    row_block_size = div(size(D, 1), n_row_devs)
    col_block_size = div(size(D, 2), n_col_devs)
    numRows  = [row_block_size for dev in 1:ndevices()]
    numCols  = [col_block_size for dev in 1:ndevices()]
    typesize = sizeof(eltype(D))
    current_dev = device()
    cpu_bufs = Vector{Matrix{eltype(D)}}(undef, ndevices())
    for (di, dev) in enumerate(devices())
        device!(dev)
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        cpu_bufs[di] = Matrix{eltype(D)}(undef, length(row_inds), length(col_inds))
        copyto!(cpu_bufs[di], dDs[di])
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        synchronize()
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 2))
        D[row_inds, col_inds] = cpu_bufs[di]
    end
    device!(current_dev)
    return D
end



## wrappers

function mg_syevd!(jobz::Char, uplo::Char, A; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevices())
    W             = Vector{real(eltype(A))}(undef, n)
    desc = MatrixDescriptor(A, grid; colblocks=N)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    IA            = 1 # for now
    JA            = 1
    GC.@preserve A_arr begin
        cusolverMgSyevd_bufferSize(mg_handle(), jobz, uplo, n, pointer.(A_arr), IA, JA, desc, W, real(eltype(A)), eltype(A), lwork)
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di] = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr workspace begin
        cusolverMgSyevd(mg_handle(), jobz, uplo, n, pointer.(A_arr), IA, JA, desc, W, real(eltype(A)), eltype(A), pointer.(workspace), lwork[], info)
    end
    chkargsok(BlasInt(info[]))
    A = returnBuffers(dev_rows, dev_cols, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), A_arr, A)
    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

function mg_potrf!(uplo::Char, A; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevices())
    desc = MatrixDescriptor(A, grid; colblocks=N)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    IA      = 1 # for now
    JA      = 1
    GC.@preserve A_arr begin
        cusolverMgPotrf_bufferSize(mg_handle(), uplo, n, pointer.(A_arr), IA, JA, desc, eltype(A), lwork)
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr workspace begin
        cusolverMgPotrf(mg_handle(), uplo, n, pointer.(A_arr), IA, JA, desc, eltype(A), pointer.(workspace), lwork[], info)
    end
    chkargsok(BlasInt(info[]))

    returnBuffers(dev_rows, dev_cols, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), A_arr, A)
end

function mg_potri!(uplo::Char, A; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    m, n    = size(A)
    N       = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevices())
    desc = MatrixDescriptor(A, grid; colblocks=N)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    IA      = 1 # for now
    JA      = 1
    GC.@preserve A_arr begin
        cusolverMgPotri_bufferSize(mg_handle(), uplo, n, pointer.(A_arr), IA, JA, desc, eltype(A), lwork)
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr workspace begin
        cusolverMgPotri(mg_handle(), uplo, n, pointer.(A_arr), IA, JA, desc, eltype(A), pointer.(workspace), lwork[], info)
    end
    chkargsok(BlasInt(info[]))

    returnBuffers(dev_rows, dev_cols, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), A_arr, A)
end

function mg_potrs!(uplo::Char, A, B; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    if uplo != 'L'
        throw(ArgumentError("only lower fill mode (uplo='L') supported"))
    end
    ma, na   = size(A)
    mb, nb   = size(B)
    NA       = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    NB       = div(size(B, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    workspace     = Vector{CuArray}(undef, ndevices())
    descA = MatrixDescriptor(A, grid; colblocks=NA)
    descB = MatrixDescriptor(A, grid; colblocks=NB)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    B_arr     = allocateBuffers(dev_rows, dev_cols, B)
    IA      = 1 # for now
    JA      = 1
    IB      = 1 # for now
    JB      = 1
    GC.@preserve A_arr B_arr begin
        cusolverMgPotrs_bufferSize(mg_handle(), uplo, na, nb, pointer.(A_arr), IA, JA, descA, pointer.(B_arr), IB, JB, descB, eltype(A), lwork)
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr B_arr workspace begin
        cusolverMgPotrs(mg_handle(), uplo, na, nb, pointer.(A_arr), IA, JA, descA, pointer.(B_arr), IB, JB, descB, eltype(A), pointer.(workspace), lwork[], info)
    end
    chkargsok(BlasInt(info[]))

    returnBuffers(dev_rows, dev_cols, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), B_arr, B)
end

function mg_getrf!(A; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    m, n    = size(A)
    N       = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    ipivs         = Vector{CuVector{Cint}}(undef, ndevices())
    workspace     = Vector{CuArray}(undef, ndevices())
    desc = MatrixDescriptor(A, grid; colblocks=N)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    IA      = 1 # for now
    JA      = 1
    for (di, dev) in enumerate(devices())
        device!(dev)
        ipivs[di]     = CUDA.zeros(Cint, N)
        synchronize()
    end
    device!(dev)
    GC.@preserve A_arr ipivs begin
        cusolverMgGetrf_bufferSize(mg_handle(), m, n, pointer.(A_arr), IA, JA, desc, pointer.(ipivs), eltype(A), lwork)
    end
    device_synchronize()
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr ipivs workspace begin
        cusolverMgGetrf(mg_handle(), m, n, pointer.(A_arr), IA, JA, desc, pointer.(ipivs), eltype(A), pointer.(workspace), lwork[], info)
    end
    device_synchronize()
    chkargsok(BlasInt(info[]))

    A = returnBuffers(dev_rows, dev_cols, div(size(A, 1), dev_rows), div(size(A, 2), dev_cols), A_arr, A)
    ipiv = Vector{Int}(undef, n)
    for (di, dev) in enumerate(devices())
        device!(dev)
        ipiv[((di-1)*N + 1):min((di*N), n)] = collect(ipivs[di])
    end
    device!(dev)
    return A, ipiv
end

function mg_getrs!(trans, A, ipiv, B; dev_rows=1, dev_cols=ndevices()) # one host-side array A
    dev = device()
    grid = DeviceGrid(1, ndevices(), devices(), CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    ma, na  = size(A)
    mb, nb  = size(B)
    NA      = div(size(A, 2), ndevices()) # dimension of the sub-matrix
    NB      = div(size(B, 2), ndevices()) # dimension of the sub-matrix
    lwork         = Ref{Int64}(0)
    ipivs         = Vector{CuVector{Cint}}(undef, ndevices())
    workspace     = Vector{CuArray}(undef, ndevices())
    descA = MatrixDescriptor(A, grid; colblocks=NA)
    descB = MatrixDescriptor(A, grid; colblocks=NB)
    A_arr     = allocateBuffers(dev_rows, dev_cols, A)
    B_arr     = allocateBuffers(dev_rows, dev_cols, B)
    IA      = 1 # for now
    JA      = 1
    IB      = 1 # for now
    JB      = 1
    for (di, dev) in enumerate(devices())
        device!(dev)
        local_ipiv    = Cint.(ipiv[(di-1)*NA+1:min(di*NA,length(ipiv))])
        ipivs[di]     = CuArray(local_ipiv)
        synchronize()
    end
    device!(dev)
    GC.@preserve A_arr B_arr ipivs begin
        cusolverMgGetrs_bufferSize(mg_handle(), trans, na, nb, pointer.(A_arr), IA, JA, descA, pointer.(ipivs), pointer.(B_arr), IB, JB, descB, eltype(A), lwork)
    end
    for (di, dev) in enumerate(devices())
        device!(dev)
        workspace[di]     = CUDA.zeros(eltype(A), lwork[])
        synchronize()
    end
    device!(dev)
    info = Ref{Cint}(C_NULL)
    GC.@preserve A_arr B_arr ipivs workspace begin
        cusolverMgGetrs(mg_handle(), trans, na, nb, pointer.(A_arr), IA, JA, descA, pointer.(ipivs), pointer.(B_arr), IB, JB, descB, eltype(A), pointer.(workspace), lwork[], info)
    end
    chkargsok(BlasInt(info[]))

    returnBuffers(dev_rows, dev_cols, div(size(B, 1), dev_rows), div(size(B, 2), dev_cols), B_arr, B)
end

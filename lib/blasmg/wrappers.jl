function cublasMgCreate()
    handle = Ref{cublasMgHandle_t}(C_NULL)
    cublasMgCreate(handle)
    return handle[]
end

function allocateBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, descs, mats)
    A, B, C  = mats
    dA, dB, dC  = descs
    a_row_block_size = div(size(A, 1), n_row_devs)
    a_col_block_size = div(size(A, 2), n_col_devs)
    b_row_block_size = div(size(B, 1), n_row_devs)
    b_col_block_size = div(size(B, 2), n_col_devs)
    c_row_block_size = div(size(C, 1), n_row_devs)
    c_col_block_size = div(size(C, 2), n_col_devs)
    a_buffers  = Vector{CuPtr{Cvoid}}(undef, num_devices)
    b_buffers  = Vector{CuPtr{Cvoid}}(undef, num_devices)
    c_buffers  = Vector{CuPtr{Cvoid}}(undef, num_devices)
    a_numRows  = Vector{Int64}(undef, num_devices)
    a_numCols  = Vector{Int64}(undef, num_devices)
    b_numRows  = Vector{Int64}(undef, num_devices)
    b_numCols  = Vector{Int64}(undef, num_devices)
    c_numRows  = Vector{Int64}(undef, num_devices)
    c_numCols  = Vector{Int64}(undef, num_devices)

    typesize = sizeof(eltype(C))
    cudaLibMgGetLocalMatrixDimensions(dA, a_numRows, a_numCols)
    cudaLibMgGetLocalMatrixDimensions(dB, b_numRows, b_numCols)
    cudaLibMgGetLocalMatrixDimensions(dC, c_numRows, c_numCols)
    ldas = Vector{Int64}(undef, num_devices)
    ldbs = Vector{Int64}(undef, num_devices)
    ldcs = Vector{Int64}(undef, num_devices)
    a_cpu_bufs = Vector{Matrix{eltype(A)}}(undef, num_devices)
    b_cpu_bufs = Vector{Matrix{eltype(B)}}(undef, num_devices)
    c_cpu_bufs = Vector{Matrix{eltype(C)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        ldas[di]    = a_numRows[di]
        ldbs[di]    = b_numRows[di]
        ldcs[di]    = c_numRows[di]
        dev_row     = mod(di - 1, n_row_devs) + 1
        dev_col     = div(di - 1, n_row_devs) + 1

        a_row_inds    = ((dev_row-1)*a_row_block_size+1):min(dev_row*a_row_block_size, size(A, 1))
        a_col_inds    = ((dev_col-1)*a_col_block_size+1):min(dev_col*a_col_block_size, size(A, 2))
        a_cpu_bufs[di] = Array(A[a_row_inds, a_col_inds])
        b_row_inds    = ((dev_row-1)*b_row_block_size+1):min(dev_row*b_row_block_size, size(B, 1))
        b_col_inds    = ((dev_col-1)*b_col_block_size+1):min(dev_col*b_col_block_size, size(B, 2))
        b_cpu_bufs[di] = Array(B[b_row_inds, b_col_inds])
        c_row_inds    = ((dev_row-1)*c_row_block_size+1):min(dev_row*c_row_block_size, size(C, 1))
        c_col_inds    = ((dev_col-1)*c_col_block_size+1):min(dev_col*c_col_block_size, size(C, 2))
        c_cpu_bufs[di] = Array(C[c_row_inds, c_col_inds])
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        if !isassigned(streams, di)
            streams[di] = CuStream()
        end
        a_gpu_buf = CuMatrix{eltype(A)}(undef, size(A))
        b_gpu_buf = CuMatrix{eltype(B)}(undef, size(B))
        c_gpu_buf = CuMatrix{eltype(C)}(undef, size(C))
        unsafe_copyto!(pointer(a_gpu_buf), pointer(a_cpu_bufs[di]), length(a_cpu_bufs[di]), stream = streams[di], async = true)
        unsafe_copyto!(pointer(b_gpu_buf), pointer(b_cpu_bufs[di]), length(b_cpu_bufs[di]), stream = streams[di], async = true)
        unsafe_copyto!(pointer(c_gpu_buf), pointer(c_cpu_bufs[di]), length(c_cpu_bufs[di]), stream = streams[di], async = true)
        a_buffers[di] = convert(CuPtr{Cvoid}, pointer(a_gpu_buf))
        b_buffers[di] = convert(CuPtr{Cvoid}, pointer(b_gpu_buf))
        c_buffers[di] = convert(CuPtr{Cvoid}, pointer(c_gpu_buf))
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
    end
    device!(deviceIdsGrid[1])
    return (a_buffers, b_buffers, c_buffers), (ldas, ldbs, ldcs)
end

function returnBuffers(grid, n_row_devs, n_col_devs, num_devices::Int, deviceIdsGrid, streams, row_block_size, col_block_size, desc, dDs, D)
    numRows  = Vector{Int64}(undef, num_devices)
    numCols  = Vector{Int64}(undef, num_devices)
    typesize = sizeof(eltype(D))
    cudaLibMgGetLocalMatrixDimensions(desc, numRows, numCols)
    current_dev = device()
    cpu_bufs = Vector{Matrix{eltype(D)}}(undef, num_devices)
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        cpu_bufs[di] = Matrix{eltype(D)}(undef, length(row_inds), length(col_inds))
        unsafe_copyto!(pointer(cpu_bufs[di]), convert(CuPtr{eltype(D)}, dDs[di]), length(cpu_bufs[di]), stream = streams[di], async = true)
    end
    for (di, dev) in enumerate(deviceIdsGrid)
        device!(dev)
        synchronize(streams[di])
        dev_row = mod(di - 1, n_row_devs) + 1
        dev_col = div(di - 1, n_row_devs) + 1
        row_inds = ((dev_row-1)*row_block_size+1):min(dev_row*row_block_size, size(D, 1))
        col_inds = ((dev_col-1)*col_block_size+1):min(dev_col*col_block_size, size(D, 1))
        D[row_inds, col_inds] = cpu_bufs[di]
    end
    device!(deviceIdsGrid[1])
    return D
end
# out of device move the memory myself
function mg_gemm_gpu!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix; devs=[0], dev_rows=1, dev_cols=1)
    device!(devs[1])
    GC.enable(false)
    grid = Ref{cudaLibMgGrid_t}(0)
    cudaLibMgCreateDeviceGrid(grid, dev_rows, dev_cols, devs, CUDALIBMG.CUDALIBMG_GRID_MAPPING_COL_MAJOR)
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    #ma, na = transA == 'N' ? size(A) : (size(A, 2), size(A, 1))
    #mb, nb = transB == 'N' ? size(B) : (size(B, 2), size(B, 1))
    ma, na = size(A)
    mb, nb = size(B)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    ldc = max(1, stride(C, 2))
    #lda = transA == 'N' ? size(A, 1) : size(A, 2) 
    #ldb = transB == 'N' ? size(B, 1) : size(B, 2) 
    #ldc = max(1, stride(C, 2))
    descA    = CudaLibMGDescriptor(A, grid[], rowblocks=div(ma, dev_rows), colblocks=div(na, dev_cols))
    descB    = CudaLibMGDescriptor(B, grid[], rowblocks=div(mb, dev_rows), colblocks=div(nb, dev_cols))
    descC    = CudaLibMGDescriptor(C, grid[], rowblocks=div(size(C, 1), dev_rows), colblocks=div(size(C, 2), dev_cols))
    ndevs    = length(devs)
    streams  = Vector{CuStream}(undef, ndevs)
    bufs, lds = allocateBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, (descA, descB, descC), (A, B, C))
    dA, dB, dC = bufs
    ldas, ldbs, ldcs = lds
    lwork     = fill(Csize_t(0x0000000100000000), ndevs)#Vector{Csize_t}(undef, ndevs)
    workspace = Vector{CUDA.Mem.DeviceBuffer}(undef, ndevs)
    workspace_ref = Vector{CuPtr{Cvoid}}(undef, ndevs)
    device!(devs[1])
    alpha_arr = [alpha]
    beta_arr  = [beta]
    cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace_ref, lwork)
    # set up workspaces and streams
    for (di, dev) in enumerate(devs)
        device!(dev)
        workspace[di] = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, lwork[di]) 
        workspace_ref[di] = workspace[di].ptr
        synchronize()
    end
    device!(devs[1])
    cublasMgGemm(mg_handle(), cutransA, cutransB, alpha_arr, descA, dA, ldas, descB, dB, ldbs, beta_arr, descC, dC, ldcs, descC, dC, ldcs, cudaDataType(eltype(C)), workspace_ref, lwork, streams)
    for (di, dev) in enumerate(devs)
        device!(dev)
        synchronize(streams[di])
        synchronize()
        CUDA.Mem.free(workspace[di])
        synchronize()
    end
    C = returnBuffers(grid, dev_rows, dev_cols, ndevs, devs, streams, div(size(C, 1), dev_rows), div(size(C, 2), dev_cols), descC, dC, C)
    GC.enable(true)
    return C
end

#=function register(A)
    buf = CUDA.Mem.register(CUDA.Mem.HostBuffer, pointer(A), sizeof(A), CUDA.Mem.HOSTREGISTER_DEVICEMAP | CUDA.Mem.HOSTREGISTER_PORTABLE)
    inalizer(A) do buf
        CUDA.Mem.unregister(buf)
    end
    return A, buf
end=#

function mg_gemm!(transA::Char,
                  transB::Char,
                  alpha::Number,
                  A::Matrix,
                  B::Matrix,
                  beta::Number,
                  C::Matrix; devs=[0], dev_rows=1, dev_cols=1)
    device!(devs[1])
    grid = CudaLibMGGrid(Int32(1), Int32(1), [Int32(-1)], CUDALIBMG_GRID_MAPPING_ROW_MAJOR)
    lda = max(1, stride(A, 2)) 
    ldb = max(1, stride(B, 2))
    ldc = max(1, stride(C, 2))
    cutransA = cublasop(transA)
    cutransB = cublasop(transB)
    a_row_blocks = size(A, 1) > 1 ? div(size(A, 1), dev_rows) : 1
    a_col_blocks = size(A, 2) > 1 ? div(size(A, 2), dev_cols) : 1
    b_row_blocks = size(B, 1) > 1 ? div(size(B, 1), dev_rows) : 1
    b_col_blocks = size(B, 2) > 1 ? div(size(B, 2), dev_cols) : 1
    c_row_blocks = size(C, 1) > 1 ? div(size(C, 1), dev_rows) : 1
    c_col_blocks = size(C, 2) > 1 ? div(size(C, 2), dev_cols) : 1
    descA    = CudaLibMGDescriptor(A, grid, rowblocks=a_row_blocks, colblocks=a_col_blocks)
    descB    = CudaLibMGDescriptor(B, grid, rowblocks=b_row_blocks, colblocks=b_col_blocks)
    descC    = CudaLibMGDescriptor(C, grid, rowblocks=c_row_blocks, colblocks=c_col_blocks)
    ndevs    = length(devs)
    C_ref_arr = Vector{Ptr{Cvoid}}(undef, ndevs)
    B_ref_arr = Vector{Ptr{Cvoid}}(undef, ndevs)
    A_ref_arr = Vector{Ptr{Cvoid}}(undef, ndevs)
    lwork     = Vector{Csize_t}(undef, ndevs)
    workspace = Vector{CUDA.Mem.DeviceBuffer}(undef, ndevs)
    workspace_ref = Vector{CUDA.CuPtr{Cvoid}}(undef, ndevs)
    streams       = Vector{CuStream}(undef, ndevs)
    CUDA.Mem.pin(A)
    CUDA.Mem.pin(B)
    CUDA.Mem.pin(C)
    #GC.@preserve descA descB descC Abuf Bbuf Cbuf A_ref_arr B_ref_arr C_ref_arr Areg Breg Creg workspace_ref lwork A B C streams begin
        for (di, dev) in enumerate(devs)
            A_ref_arr[di] = Base.unsafe_convert(Ptr{Cvoid}, pointer(A))
            B_ref_arr[di] = Base.unsafe_convert(Ptr{Cvoid}, pointer(B))
            C_ref_arr[di] = Base.unsafe_convert(Ptr{Cvoid}, pointer(C))
        end
        device!(devs[1])
        ldcc      = [Int64(ldc)]
        ldaa      = [Int64(lda)]
        ldbb      = [Int64(ldb)]
        cublasMgGemmWorkspace(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, ldaa, descB, B_ref_arr, ldbb, [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace_ref, lwork)
        # set up workspaces and streams
        for (di, dev) in enumerate(devs)
            device!(dev)
            workspace[di] = CUDA.Mem.alloc(CUDA.Mem.DeviceBuffer, lwork[di])
            workspace_ref[di] = workspace[di].ptr 
            streams[di]   = CuDefaultStream()
            synchronize(streams[di])
            synchronize()
        end
        device!(devs[1])
        cublasMgGemm(mg_handle(), cutransA, cutransB, [alpha], descA, A_ref_arr, ldaa, descB, B_ref_arr, ldbb, [beta], descC, C_ref_arr, ldcc, descC, C_ref_arr, ldcc, cudaDataType(eltype(C)), workspace_ref, lwork, streams)
        for (di, dev) in enumerate(devs)
            device!(dev)
            synchronize(streams[di])
            synchronize()
            CUDA.Mem.free(workspace[di])
        end
        device!(devs[1])
        #CUDA.Mem.unregister(Abuf)
        #CUDA.Mem.unregister(Bbuf)
        #CUDA.Mem.unregister(Cbuf)
    #end
    return C
end

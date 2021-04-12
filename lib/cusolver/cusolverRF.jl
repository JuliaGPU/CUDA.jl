
import CUDA.CUBLAS: unsafe_batch, unsafe_strided_batch

function cusolverRfCreate()
    handle_ref = Ref{cusolverRfHandle_t}()
    cusolverRfCreate(handle_ref)
    return handle_ref[]
end

function cusolverRfFree(handle)
    if handle != C_NULL
        cusolverRfDestroy(handle)
        handle = C_NULL
    end
end

mutable struct RfHandle
    handle::Ptr{cusolverRfHandle_t}
end

function sparse_rf_handle(;
    fast_mode=true, nzero=0.0, nboost=0.0,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
)
    # Create handle
    gH = cusolverRfCreate()
    if fast_mode
        cusolverRfSetResetValuesFastMode(gH, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON)
    else
        cusolverRfSetResetValuesFastMode(gH, CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF)
    end
    cusolverRfSetNumericProperties(gH, nzero, nboost)
    cusolverRfSetMatrixFormat(
        gH,
        CUSOLVERRF_MATRIX_FORMAT_CSR,
        CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L
    )
    cusolverRfSetAlgs(
        gH, factorization_algo, triangular_algo,
    )
    handle = RfHandle(gH)
    finalizer(rf_free!, handle)
    return handle
end

rf_free!(rf::RfHandle) = cusolverRfFree(rf.handle)

Base.unsafe_convert(::Type{cusolverRfHandle_t}, rf::RfHandle) = rf.handle

struct RfHostLU{T}
    nnzA::Cint
    rowsA::Vector{Cint}
    colsA::Vector{Cint}
    valsA::Vector{T}
    nnzL::Cint
    rowsL::Vector{Cint}
    colsL::Vector{Cint}
    valsL::Vector{T}
    nnzU::Cint
    rowsU::Vector{Cint}
    colsU::Vector{Cint}
    valsU::Vector{T}
    P::Vector{Cint}
    Q::Vector{Cint}
end

function RfHostLU(
    A::CuSparseMatrixCSR{T, Ti};
    ordering=:AMD, tol=1e-8, check=true,
) where {T, Ti}
    m, n = size(A)
    @assert m == n # only squared matrices are supported
    nnzA = nnz(A)

    # Transfer data to host
    h_rowsA = A.rowPtr |> Vector{Cint}
    h_colsA = A.colVal |> Vector{Cint}
    h_valsA = A.nzVal |> Vector{T}

    # cusolverRf is 0-based
    h_rowsA .-= Cint(1)
    h_colsA .-= Cint(1)
    h_Qreorder = zeros(Cint, n)
    # Create duplicate matrix for reordering
    h_rowsB = copy(h_rowsA)
    h_colsB = copy(h_colsA)
    h_valsB = copy(h_valsA)

    spH = sparse_handle()

    # Create matrix descriptor
    desca = CUSPARSE.CuMatrixDescriptor()
    CUSPARSE.cusparseSetMatType(desca, CUSPARSE.CUSPARSE_MATRIX_TYPE_GENERAL)
    CUSPARSE.cusparseSetMatIndexBase(desca, CUSPARSE.CUSPARSE_INDEX_BASE_ZERO)

    # Reordering
    if ordering == :AMD
        cusolverSpXcsrsymamdHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    elseif ordering == :MDQ
        cusolverSpXcsrsymmdqHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    elseif ordering == :METIS
        cusolverSpXcsrmetisndHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, C_NULL, h_Qreorder,
        )
    elseif ordering == :RCM
        cusolverSpXcsrsymrcmHost(
            spH,
            n, nnzA, desca,
            h_rowsA, h_colsA, h_Qreorder,
        )
    end

    h_mapBfromA = zeros(Cint, nnzA)
    @inbounds for i in 1:nnzA
        h_mapBfromA[i] = i # identity matrix
    end

    # Compute permutation in two steps
    size_perm = Ref{Csize_t}(0)
    cusolverSpXcsrperm_bufferSizeHost(
        spH,
        m, n, nnzA, desca,
        h_rowsB, h_colsB, h_Qreorder, h_Qreorder,
        size_perm,
    )

    buffer_cpu = zeros(Cint, size_perm[])
    cusolverSpXcsrpermHost(
        spH,
        m, n, nnzA, desca,
        h_rowsB, h_colsB, h_Qreorder, h_Qreorder, h_mapBfromA,
        buffer_cpu,
    )

    # Apply permutation
    h_valsB = h_valsA[h_mapBfromA]

    # LU Factorization
    info = Ref{CUSOLVER.csrqrInfo_t}()
    cusolverSpCreateCsrluInfoHost(info)

    cusolverSpXcsrluAnalysisHost(
        spH,
        m, nnzA, desca,
        h_rowsB, h_colsB, info[],
    )

    size_internal = Ref{Cint}(0)
    size_lu = Ref{Cint}(0)
    cusolverSpDcsrluBufferInfoHost(
        spH,
        n, nnzA, desca,
        h_valsB, h_rowsB, h_colsB,
        info[],
        size_internal, size_lu
    )

    n_bytes = size_lu[] * sizeof(Cint)
    buffer_lu = zeros(Cint, size_lu[])
    pivot_threshold = 1.0

    cusolverSpDcsrluFactorHost(
        spH, n, nnzA, desca,
        h_valsB, h_rowsB, h_colsB,
        info[], pivot_threshold,
        buffer_lu,
    )

    # Check singularity
    if check
        singularity = Ref{Cint}(0)
        cusolverSpDcsrluZeroPivotHost(
            spH, info[], tol, singularity,
        )

        # Check that the matrix is nonsingular
        if singularity[] >= 0
            SingularException(singularity[])
        end
    end

    # Get size of L and U
    pnnzU = Ref{Cint}(0)
    pnnzL = Ref{Cint}(0)
    cusolverSpXcsrluNnzHost(
        spH,
        pnnzL, pnnzU, info[],
    )

    nnzL = pnnzL[]
    nnzU = pnnzU[]

    # Retrieve L and U matrices
    h_Plu = zeros(Cint, m)
    h_Qlu = zeros(Cint, n)

    h_valsL = zeros(nnzL)
    h_rowsL = zeros(Cint, m+1)
    h_colsL = zeros(Cint, nnzL)

    h_valsU = zeros(nnzU)
    h_rowsU = zeros(Cint, m+1)
    h_colsU = zeros(Cint, nnzU)

    # Extract
    cusolverSpDcsrluExtractHost(
        spH,
        h_Plu, h_Qlu,
        desca,
        h_valsL, h_rowsL, h_colsL,
        desca,
        h_valsU, h_rowsU, h_colsU,
        info[],
        buffer_lu,
    )

    h_P = h_Qreorder[h_Plu .+ 1]
    h_Q = h_Qreorder[h_Qlu .+ 1]

    return RfHostLU(
        nnzA, h_rowsA, h_colsA, h_valsA,
        nnzL, h_rowsL, h_colsL, h_valsL,
        nnzU, h_rowsU, h_colsU, h_valsU,
        h_P, h_Q,
    )
end

struct RfLU{T} <: LinearAlgebra.Factorization{T}
    rf::RfHandle
    nrhs::Int
    n::Int
    m::Int
    nnzA::Int
    drowsA::CuVector{Cint}
    dcolsA::CuVector{Cint}
    dP::CuVector{Cint}
    dQ::CuVector{Cint}
    dT::CuVector{T}
end

function RfLU(
    A::CuSparseMatrixCSR{T, Ti};
    nrhs=1, ordering=:AMD, check=true, fast_mode=true,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
) where {T, Ti}
    if nrhs > 1
        error("Currently CusolverRF supports only one right-hand side.")
    end
    n, m = size(A)
    lu_host = RfHostLU(A; ordering=ordering, check=check)

    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * nrhs)

    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Assemble internal data structures
    cusolverRfSetupHost(
        n, lu_host.nnzA, lu_host.rowsA, lu_host.colsA, lu_host.valsA,
        lu_host.nnzL, lu_host.rowsL, lu_host.colsL, lu_host.valsL,
        lu_host.nnzU, lu_host.rowsU, lu_host.colsU, lu_host.valsU,
        lu_host.P, lu_host.Q,
        rf
    )
    # Analyze available parallelism
    cusolverRfAnalyze(rf)
    # LU factorization
    cusolverRfRefactor(rf)

    return RfLU{T}(
        rf, nrhs, n, m, lu_host.nnzA,
        lu_host.rowsA, lu_host.colsA, lu_host.P, lu_host.Q, d_T
    )
end

# Update factorization inplace
function rf_refactor!(rflu::RfLU{T}, A::CuSparseMatrixCSR{T, Ti}) where {T, Ti}
    cusolverRfResetValues(
        rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, A.nzVal, rflu.dP, rflu.dQ,
        rflu.rf
    )
    cusolverRfRefactor(rflu.rf)
    return
end

# Solve system Ax = b
function rf_solve!(rflu::RfLU{T}, x::CuVector{T}) where T
    n = rflu.n
    cusolverRfSolve(rflu.rf, rflu.dP, rflu.dQ, rflu.nrhs, rflu.dT, n, x, n)
    return
end

# Batch factorization should not mix with classical LU factorization.
# We implement a structure apart.
struct RfBatchLU{T} <: LinearAlgebra.Factorization{T}
    rf::RfHandle
    batchsize::Int
    n::Int
    m::Int
    nnzA::Int
    drowsA::CuVector{Cint}
    dcolsA::CuVector{Cint}
    dP::CuVector{Cint}
    dQ::CuVector{Cint}
    dT::CuVector{T}
end

function RfBatchLU(
    A::CuSparseMatrixCSR{T, Ti}, batchsize::Int;
    ordering=:AMD, check=true, fast_mode=true,
    factorization_algo=CUSOLVERRF_FACTORIZATION_ALG0,
    triangular_algo=CUSOLVERRF_TRIANGULAR_SOLVE_ALG1,
) where {T, Ti}
    n, m = size(A)
    lu_host = RfHostLU(A; ordering=ordering, check=check)

    # Allocations (device)
    d_T = CUDA.zeros(Cdouble, m * batchsize * 2)

    rf = sparse_rf_handle(;
        fast_mode=fast_mode,
        factorization_algo=factorization_algo,
        triangular_algo=triangular_algo,
    )

    # Assemble internal data structures
    h_valsA_batch = Vector{Float64}[lu_host.valsA for i in 1:batchsize]
    ptrA_batch = pointer.(h_valsA_batch)
    cusolverRfBatchSetupHost(
        batchsize,
        n, lu_host.nnzA, lu_host.rowsA, lu_host.colsA, ptrA_batch,
        lu_host.nnzL, lu_host.rowsL, lu_host.colsL, lu_host.valsL,
        lu_host.nnzU, lu_host.rowsU, lu_host.colsU, lu_host.valsU,
        lu_host.P, lu_host.Q,
        rf,
    )
    # Analyze available parallelism
    cusolverRfBatchAnalyze(rf)
    # LU factorization
    cusolverRfBatchRefactor(rf)

    return RfBatchLU{T}(
        rf, batchsize, n, m, lu_host.nnzA,
        lu_host.rowsA, lu_host.colsA, lu_host.P, lu_host.Q, d_T
    )
end

# Update factorization inplace
## Single matrix
function rf_batch_refactor!(rflu::RfBatchLU{T}, A::CuSparseMatrixCSR{T, Ti}) where {T, Ti}
    ptrs = [pointer(A.nzVal) for i in 1:rflu.batchsize]
    Aptrs = CuArray(ptrs)
    cusolverRfBatchResetValues(
        rflu.batchsize, rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, Aptrs, rflu.dP, rflu.dQ,
        rflu.rf
    )
    unsafe_free!(Aptrs)
    cusolverRfBatchRefactor(rflu.rf)
    return
end
## Multiple matrices
function rf_batch_refactor!(rflu::RfBatchLU{T}, As::Vector{CuSparseMatrixCSR{T, Ti}}) where {T, Ti}
    @assert length(As) == rflu.batchsize
    ptrs = [pointer(A.nzVal) for A in As]
    Aptrs = CuArray(ptrs)
    cusolverRfBatchResetValues(
        rflu.batchsize, rflu.n, rflu.nnzA,
        rflu.drowsA, rflu.dcolsA, Aptrs, rflu.dP, rflu.dQ,
        rflu.rf
    )
    unsafe_free!(Aptrs)
    cusolverRfBatchRefactor(rflu.rf)
    return
end

function rf_batch_solve!(rflu::RfBatchLU{T}, xs::Vector{CuVector{T}}) where T
    @assert length(xs) == rflu.batchsize
    n, nrhs = rflu.n, 1
    Xptrs = unsafe_batch(xs)
    cusolverRfBatchSolve(rflu.rf, rflu.dP, rflu.dQ, nrhs, rflu.dT, n, Xptrs, n)
    unsafe_free!(Xptrs)
    return
end

function rf_batch_solve!(rflu::RfBatchLU{T}, X::CuMatrix{T}) where T
    @assert size(X, 2) == rflu.batchsize
    n = rflu.n
    nrhs = 1
    Xptrs = unsafe_strided_batch(X)
    # Forward and backward solve
    cusolverRfBatchSolve(rflu.rf, rflu.dP, rflu.dQ, nrhs, rflu.dT, n, Xptrs, n)
    unsafe_free!(Xptrs)
    return
end

# Operators overloading
function LinearAlgebra.ldiv!(x::CuArray{T}, rflu::RfLU{T}, b::CuArray{T}) where T
    copyto!(x, b)
    rf_solve!(rflu, x)
end
function LinearAlgebra.ldiv!(rflu::RfLU{T}, x::CuArray{T}) where T
    rf_solve!(rflu, x)
end

LinearAlgebra.lu(A::CuSparseMatrixCSR; options...) = RfLU(A; options...)
LinearAlgebra.lu!(rflu::RfLU, A::CuSparseMatrixCSR) = rf_refactor!(rflu, A)

# Batch
function LinearAlgebra.ldiv!(x::CuMatrix{T}, rflu::RfBatchLU{T}, b::CuMatrix{T}) where T
    copyto!(x, b)
    rf_batch_solve!(rflu, x)
end
function LinearAlgebra.ldiv!(rflu::RfBatchLU{T}, x::CuMatrix{T}) where T
    rf_batch_solve!(rflu, x)
end

LinearAlgebra.lu!(rflu::RfBatchLU, A::CuSparseMatrixCSR) = rf_batch_refactor!(rflu, A)


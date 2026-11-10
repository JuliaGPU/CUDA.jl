# The wrappers in this file use cuSOLVER's 64-bit generic API (cusolverDnX*), which is
# available from cuSOLVER 11. Where the legacy 32-bit API offers the same routine, it is
# used on older versions.

# potrf
function potrf!(uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_potrf!(uplo, A)
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXpotrf_bufferSize(dh, params, uplo, n,
                                    T, A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXpotrf(dh, params, uplo, n, T, A, lda, T,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, flag
end

# potrs
function potrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_potrs!(uplo, A, B)
    chkuplo(uplo)
    n = checksquare(A)
    p, nrhs = size(B, 1), size(B, 2)
    (p ≠ n) && throw(DimensionMismatch("first dimension of B, $p, must match second dimension of A, $n"))
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXpotrs(dh, params, uplo, n, nrhs, T, A, lda, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# getrf (the methods taking 32-bit pivots use the legacy API, see dense.jl)
function getrf!(A::StridedCuMatrix{T}, ipiv::CuVector{Int64}) where {T <: BlasFloat}
    if version() < v"11"
        ipiv32 = CuVector{Cint}(undef, length(ipiv))
        _, _, info = getrf!(A, ipiv32)
        ipiv .= ipiv32
        return A, ipiv, info
    end
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgetrf_bufferSize(dh, params, m, n, T,
                                    A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgetrf(dh, params, m, n, T, A, lda, ipiv,
                         T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, ipiv, flag
end

# getrs
function getrs!(trans::Char, A::StridedCuMatrix{T}, ipiv::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && return getrs!(trans, A, CuVector{Cint}(ipiv), B)
    # support trans = 'C' for real matrices
    trans = T <: Real && trans == 'C' ? 'T' : trans
    chktrans(trans)
    n = checksquare(A)
    if size(B, 1) != n
        throw(DimensionMismatch("first dimension of B, $(size(B,1)), must match dimension of A, $n"))
    end
    if length(ipiv) != n
        throw(DimensionMismatch("length of ipiv, $(length(ipiv)), must match dimension of A, $n"))
    end
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    cusolverDnXgetrs(dh, params, trans, n, nrhs, T, A, lda, ipiv, T, B, ldb, dh.info)

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# geqrf
function geqrf!(A::StridedCuMatrix{T}, tau::CuVector{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_geqrf!(A, tau)
    m, n = size(A)
    lda = max(1, stride(A, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeqrf_bufferSize(dh, params, m, n, T, A,
                                    lda, T, tau, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeqrf(dh, params, m, n, T, A,
                         lda, T, tau, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A, tau
end

function geqrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    tau = CuVector{T}(undef, min(m,n))
    geqrf!(A, tau)
end

# Xsytrs
function sytrs!(uplo::Char, A::StridedCuMatrix{T}, p::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && error("This operation requires cuSOLVER 11 or later.")
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, p, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, p,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

function sytrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    version() < v"11" && error("This operation requires cuSOLVER 11 or later.")
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dh, uplo, n, nrhs, T, A,
                                    lda, CU_NULL, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu,
                    bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dh, uplo, n, nrhs, T, A, lda, CU_NULL,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    B
end

# Xtrtri
function trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && error("This operation requires cuSOLVER 11 or later.")
    chkuplo(uplo)
    chkdiag(diag)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXtrtri_bufferSize(dh, uplo, diag, n, T, A, lda, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXtrtri(dh, uplo, diag, n, T, A, lda,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu),
                         dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)
    A
end

# Xlarft!
function larft!(direct::Char, storev::Char, v::StridedCuMatrix{T}, tau::StridedCuVector{T}, t::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.6.0" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n, k = size(v)
    ktau = length(tau)
    mt, nt = size(t)
    (storev != 'C') && throw(ArgumentError("Only storev = 'C' is supported."))
    (n < k) && throw(ArgumentError("The number of elementary reflectors ($k) must be lower or equal to the order of block reflector H ($n)."))
    (ktau != k) && throw(ArgumentError("The length of tau ($ktau) is not equal to the number of elementary reflectors ($k)."))
    (mt != k || nt != k) && throw(ArgumentError("The size of the triangular factor of the block reflector is ($mt, $nt) and must be ($k, $k)."))
    ldv = max(1, stride(v, 2))
    ldt = max(1, stride(t, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXlarft_bufferSize(dh, params, direct, storev, n, k, T,
                                    v, ldv, T, tau, T, t, ldt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXlarft(dh, params, direct, storev, n, k, T, v, ldv, T, tau, T, t,
                         ldt, T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu))
    end

    t
end

# gesvd
function gesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_gesvd!(jobu, jobvt, A)
    m, n = size(A)
    R = real(T)
    (m < n) && throw(ArgumentError("The number of rows of A ($m) must be greater or equal to the number of columns of A ($n)"))
    k = min(m, n)
    U = if jobu == 'A'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'S'
        CuMatrix{T}(undef, m, k)
    elseif jobu == 'N' || jobu == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    Σ = CuVector{R}(undef, k)
    Vt = if jobvt == 'A'
        CuMatrix{T}(undef, n, n)
    elseif jobvt == 'S'
        CuMatrix{T}(undef, k, n)
    elseif jobvt == 'N' || jobvt == 'O'
        CU_NULL
    else
        throw(ArgumentError("jobvt is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldvt = Vt == CU_NULL ? 1 : max(1, stride(Vt, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvd_bufferSize(dh, params, jobu, jobvt,
                                    m, n, T, A, lda, R, Σ, T, U, ldu,
                                    T, Vt, ldvt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvd(dh, params, jobu, jobvt, m, n, T, A,
                         lda, R, Σ, T, U, ldu, T, Vt, ldvt, T, buffer_gpu,
                         sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, Vt
end

# gesvdp
function gesvdp!(jobz::Char, econ::Bool, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    p = min(m, n)
    R = real(T)
    jobz in ('N', 'V') || throw(ArgumentError("jobz is incorrect. The values accepted are 'V' and 'N'."))

    U = CuMatrix{T}(undef, m, econ ? p : m)
    V = CuMatrix{T}(undef, n, econ ? p : n)
    Σ = CuVector{R}(undef, p)
    lda = max(1, stride(A, 2))
    ldu = max(1, stride(U, 2))
    ldv = max(1, stride(V, 2))
    h_err_sigma = Ref{Cdouble}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdp_bufferSize(dh, params, jobz, econ, m,
                                     n, T, A, lda, R, Σ, T, U, ldu, T, V,
                                     ldv, T, out_gpu, out_cpu)

        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdp(dh, params, jobz, econ, m, n, T, A, lda, R,
                          Σ, T, U, ldu, T, V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info, h_err_sigma)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    if jobz == 'N'
        unsafe_free!(U)
        unsafe_free!(V)
        return Σ, h_err_sigma[]
    elseif jobz == 'V'
        return U, Σ, V, h_err_sigma[]
    end
end

# gesvdr
function gesvdr!(jobu::Char, jobv::Char, A::StridedCuMatrix{T}, k::Integer;
                  niters::Integer=2, p::Integer=2*k) where {T <: BlasFloat}
    m, n = size(A)
    ℓ = min(m,n)
    p = min(p, ℓ-k)  # Ensure that p + k ≤ ℓ
    (1 ≤ k ≤ ℓ) || throw(ArgumentError("illegal choice of parameter k = $k, which must be between 1 and min(m,n) = $ℓ"))
    (k+p ≤ ℓ) || throw(ArgumentError("illegal choice of parameters k = $k and p = $p, which must satisfy k+p ≤ min(m,n) = $ℓ"))
    R = real(T)
    U = if jobu == 'S'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'N'
        CuMatrix{T}(undef, m, ℓ)
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'S' and 'N'."))
    end
    Σ = CuVector{R}(undef, ℓ)
    V = if jobv == 'S'
        CuMatrix{T}(undef, n, n)
    elseif jobv == 'N'
        CuMatrix{T}(undef, n, ℓ)
    else
        throw(ArgumentError("jobv is incorrect. The values accepted are 'S' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldv = V == CU_NULL ? 1 : max(1, stride(V, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdr_bufferSize(dh, params, jobu, jobv,
                                     m, n, k, p, niters, T, A, lda, R, Σ, T,
                                     U, ldu, T, V, ldv, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdr(dh, params, jobu, jobv, m, n,
                          k, p, niters, T, A, lda, R, Σ, T, U, ldu, T,
                          V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chklapackerror(flag |> BlasInt)
    U, Σ, V
end

# syevd (heevd for complex matrices)
function syevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    version() < v"11" && return legacy_syevd!(jobz, uplo, A)
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevd_bufferSize(dh, params, jobz, uplo, n,
                                    T, A, lda, R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevd(dh, params, jobz, uplo, n, T, A,
                         lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# syevdx
function syevdx!(jobz::Char, range::Char, uplo::Char, A::StridedCuMatrix{T};
                  vl::Real=0.0, vu::Real=Inf, il::Integer=1, iu::Integer=0) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    iu = iu == 0 ? n : iu
    (range == 'I') && !(1 ≤ il ≤ iu ≤ n) && throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
    (range == 'V') && (vl ≥ vu) && throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
    lda = max(1, stride(A, 2))
    W = CuVector{R}(undef, n)
    vl = Ref{R}(vl)
    vu = Ref{R}(vu)
    h_meig = Ref{Int64}(0)
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevdx_bufferSize(dh, params, jobz, range, uplo, n,
                                     T, A, lda, vl, vu, il, iu, h_meig,
                                     R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevdx(dh, params, jobz, range, uplo, n, T, A,
                          lda, vl, vu, il, iu, h_meig, R, W, T, buffer_gpu,
                          sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    if jobz == 'N'
        return W, h_meig[]
    elseif jobz == 'V'
        return W, A, h_meig[]
    end
end

# geev
function geev!(jobvl::Char, jobvr::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    cuSOLVER.version() < v"11.7.1" && throw(ErrorException("This operation is not supported by the current CUDA version."))
    n = checksquare(A)
    VL = if jobvl == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvl == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvl is incorrect. The values accepted are 'V' and 'N'."))
    end
    C = T <: Real ? Complex{T} : T
    W = CuVector{C}(undef, n)
    VR = if jobvr == 'V'
        CuMatrix{T}(undef, n, n)
    elseif jobvr == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvr is incorrect. The values accepted are 'V' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldvl = VL == CU_NULL ? 1 : max(1, stride(VL, 2))
    ldvr = VR == CU_NULL ? 1 : max(1, stride(VR, 2))
    params = CuSolverParameters()
    dh = dense_handle()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeev_bufferSize(dh, params, jobvl, jobvr, n, T, A,
                                   lda, C, W, T, VL, ldvl, T, VR, ldvr,
                                   T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeev(dh, params, jobvl, jobvr, n, T, A, lda, C,
                        W, T, VL, ldvl, T, VR, ldvr, T, buffer_gpu,
                        sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), dh.info)
    end

    flag = @allowscalar dh.info[1]
    chkargsok(flag |> BlasInt)

    return W, VL, VR
end

# syevBatched
function syevBatched!(jobz::Char, uplo::Char, A::StridedCuArray{T, 3}) where {T <: BlasFloat}
    minimum_version = v"11.7.1"
    cuSOLVER.version() < minimum_version && throw(ErrorException("This operation requires cuSOLVER
        $(minimum_version) or later. Current cuSOLVER version: $(cuSOLVER.version())."))
    chkuplo(uplo)
    n = checksquare(A)
    batch_size = size(A, 3)
    R = real(T)
    lda = max(1, stride(A, 2))
    W = CuMatrix{R}(undef, n, batch_size)
    params = CuSolverParameters()
    dh = dense_handle()
    resize!(dh.info, batch_size)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevBatched_bufferSize(
            dh, params, jobz, uplo, n,
            T, A, lda, R, W, T, out_gpu, out_cpu, batch_size
        )
        return out_gpu[], out_cpu[]
    end
    with_workspaces(dh.workspace_gpu, dh.workspace_cpu, bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevBatched(
            dh, params, jobz, uplo, n, T, A,
            lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
            buffer_cpu, sizeof(buffer_cpu), dh.info, batch_size
        )
    end

    info = @allowscalar collect(dh.info)
    for i in 1:batch_size
        chkargsok(info[i] |> BlasInt)
    end

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LinearAlgebra.LAPACK.sytrs!(uplo::Char, A::StridedCuMatrix{$elty}, p::CuVector{Int64}, B::StridedCuVecOrMat{$elty}) = cuSOLVER.sytrs!(uplo, A, p, B)
        LinearAlgebra.LAPACK.trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{$elty}) = cuSOLVER.trtri!(uplo, diag, A)
    end
end

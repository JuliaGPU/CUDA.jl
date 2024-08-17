mutable struct CuSolverParameters
    parameters::cusolverDnParams_t

    function CuSolverParameters()
        parameters_ref = Ref{cusolverDnParams_t}()
        cusolverDnCreateParams(parameters_ref)
        obj = new(parameters_ref[])
        finalizer(cusolverDnDestroyParams, obj)
        obj
    end
end

Base.unsafe_convert(::Type{cusolverDnParams_t}, params::CuSolverParameters) = params.parameters

# Xpotrf
function Xpotrf!(uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXpotrf_bufferSize(dense_handle(), params, uplo, n,
                                    T, A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXpotrf(dense_handle(), params, uplo, n, T, A, lda, T,
                         buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    A, flag
end

# Xpotrs
function Xpotrs!(uplo::Char, A::StridedCuMatrix{T}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    p, nrhs = size(B)
    (p ≠ n) && throw(DimensionMismatch("first dimension of B, $p, must match second dimension of A, $n"))
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    cusolverDnXpotrs(dense_handle(), params, uplo, n, nrhs, T, A, lda, T, B, ldb, info)

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    B
end

# Xgetrf
function Xgetrf!(A::StridedCuMatrix{T}, ipiv::CuVector{Int64}) where {T <: BlasFloat}
    m, n = size(A)
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgetrf_bufferSize(dense_handle(), params, m, n, T,
                                    A, lda, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgetrf(dense_handle(), params, m, n, T, A, lda, ipiv,
                         T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu,
                         sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    A, ipiv, flag
end

function Xgetrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m,n = size(A)
    ipiv = CuVector{Int64}(undef, min(m, n))
    Xgetrf!(A, ipiv)
end

# Xgetrs
function Xgetrs!(trans::Char, A::StridedCuMatrix{T}, ipiv::CuVector{Int64}, B::StridedCuVecOrMat{T}) where {T <: BlasFloat}
    chktrans(trans)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    cusolverDnXgetrs(dense_handle(), params, trans, n, nrhs, T, A, lda, ipiv, T, B, ldb, info)

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    B
end

# Xgeqrf
function Xgeqrf!(A::StridedCuMatrix{T}, tau::CuVector{T}) where {T <: BlasFloat}
    m, n = size(A)
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgeqrf_bufferSize(dense_handle(), params, m, n, T, A,
                                    lda, T, tau, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgeqrf(dense_handle(), params, m, n, T, A,
                         lda, T, tau, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    A, tau
end

function Xgeqrf!(A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    tau = CuVector{T}(undef, min(m,n))
    Xgeqrf!(A, tau)
end

# Xsytrs
function sytrs!(uplo::Char, A::StridedCuMatrix{T}, p::CuVector{Int64}, B::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    nrhs = size(B, 2)
    lda = max(1, stride(A, 2))
    ldb = max(1, stride(B, 2))
    info = CuVector{Cint}(undef, 1)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsytrs_bufferSize(dense_handle(), uplo, n, nrhs, T, A,
                                    lda, p, T, B, ldb, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsytrs(dense_handle(), uplo, n, nrhs, T, A, lda, p,
                         T, B, ldb, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    B
end

# Xtrtri
function trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    chkdiag(diag)
    n = checksquare(A)
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXtrtri_bufferSize(dense_handle(), uplo, diag, n, T, A, lda, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXtrtri(dense_handle(), uplo, diag, n, T, A, lda, buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))
    A
end

# Xlarft!
function larft!(direct::Char, storev::Char, v::StridedCuMatrix{T}, tau::StridedCuVector{T}, t::StridedCuMatrix{T}) where {T <: BlasFloat}
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

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXlarft_bufferSize(dense_handle(), params, direct, storev, n, k, T,
                                    v, ldv, T, tau, T, t, ldt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXlarft(dense_handle(), params, direct, storev, n, k, T, v, ldv, T, tau, T, t,
                         ldt, T, buffer_gpu, sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu))
    end

    t
end

# Xgesvd
function Xgesvd!(jobu::Char, jobvt::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    R = real(T)
    (m < n) && throw(ArgumentError("The number of rows of A ($m) must be greater or equal to the number of columns of A ($n)"))
    U = if jobu == 'A'
        CuMatrix{T}(undef, m, m)
    elseif jobu == 'S' || jobu == 'O'
        CuMatrix{T}(undef, m, min(m, n))
    elseif jobu == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    Σ = CuVector{R}(undef, min(m, n))
    Vt = if jobvt == 'A'
        CuMatrix{T}(undef, n, n)
    elseif jobvt == 'S' || jobvt == 'O'
        CuMatrix{T}(undef, min(m, n), n)
    elseif jobvt == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobvt is incorrect. The values accepted are 'A', 'S', 'O' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldvt = Vt == CU_NULL ? 1 : max(1, stride(Vt, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvd_bufferSize(dense_handle(), params, jobu, jobvt,
                                    m, n, T, A, lda, R, Σ, T, U, ldu,
                                    T, Vt, ldvt, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvd(dense_handle(), params, jobu, jobvt, m, n, T, A,
                         lda, R, Σ, T, U, ldu, T, Vt, ldvt, T, buffer_gpu,
                         sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chklapackerror(BlasInt(flag))
    U, Σ, Vt
end

# Xgesvdp
function Xgesvdp!(jobz::Char, econ::Int, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    m, n = size(A)
    p = min(m, n)
    R = real(T)
    econ ∈ (0, 1) || throw(ArgumentError("econ is incorrect. The values accepted are 0 and 1."))
    U = if jobz == 'V' && econ == 1
        CuMatrix{T}(undef, m, p)
    elseif jobz == 'V' && econ == 0
        CuMatrix{T}(undef, m, m)
    elseif jobz == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobz is incorrect. The values accepted are 'V' and 'N'."))
    end
    Σ = CuVector{R}(undef, p)
    V = if jobz == 'V' && econ == 1
        CuMatrix{T}(undef, n, p)
    elseif jobz == 'V' && econ == 0
        CuMatrix{T}(undef, n, n)
    elseif jobz == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobz is incorrect. The values accepted are 'V' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldv = V == CU_NULL ? 1 : max(1, stride(V, 2))
    info = CuVector{Cint}(undef, 1)
    h_err_sigma = Ref{Cdouble}(0)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdp_bufferSize(dense_handle(), params, jobz, econ, m,
                                     n, T, A, lda, R, Σ, T, U, ldu, T, V,
                                     ldv, T, out_gpu, out_cpu)

        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdp(dense_handle(), params, jobz, econ, m, n, T, A, lda, R,
                          Σ, T, U, ldu, T, V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), info, h_err_sigma)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chklapackerror(BlasInt(flag))
    if jobz == 'N'
        return Σ, h_err_sigma[]
    elseif jobz == 'V'
        return U, Σ, V, h_err_sigma[]
    end
end

# Xgesvdr
function Xgesvdr!(jobu::Char, jobv::Char, A::StridedCuMatrix{T}, k::Integer;
                  niters::Integer=2, p::Integer=2*k) where {T <: BlasFloat}
    m, n = size(A)
    ℓ = min(m,n)
    (1 ≤ k ≤ ℓ) || throw(ArgumentError("illegal choice of parameter k = $k, which must be between 1 and min(m,n) = $ℓ"))
    (k+p ≤ ℓ) || throw(ArgumentError("illegal choice of parameters k = $k and p = $p, which must satisfy k+p ≤ min(m,n) = $ℓ"))
    R = real(T)
    U = if jobu == 'S'
        CuMatrix{T}(undef, m, k)
    elseif jobu == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobu is incorrect. The values accepted are 'S' and 'N'."))
    end
    Σ = CuVector{R}(undef, ℓ)
    V = if jobv == 'S'
        CuMatrix{T}(undef, n, k)
    elseif jobv == 'N'
        CU_NULL
    else
        throw(ArgumentError("jobv is incorrect. The values accepted are S' and 'N'."))
    end
    lda = max(1, stride(A, 2))
    ldu = U == CU_NULL ? 1 : max(1, stride(U, 2))
    ldv = V == CU_NULL ? 1 : max(1, stride(V, 2))
    info = CuVector{Cint}(undef, 1)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXgesvdr_bufferSize(dense_handle(), params, jobu, jobv,
                                     m, n, k, p, niters, T, A, lda, R, Σ, T,
                                     U, ldu, T, V, ldv, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXgesvdr(dense_handle(), params, jobu, jobv, m, n,
                          k, p, niters, T, A, lda, R, Σ, T, U, ldu, T,
                          V, ldv, T, buffer_gpu, sizeof(buffer_gpu),
                          buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chklapackerror(BlasInt(flag))
    U, Σ, V
end

# Xsyevd
function Xsyevd!(jobz::Char, uplo::Char, A::StridedCuMatrix{T}) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)
    W = CuVector{R}(undef, n)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevd_bufferSize(dense_handle(), params, jobz, uplo, n,
                                    T, A, lda, R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevd(dense_handle(), params, jobz, uplo, n, T, A,
                         lda, R, W, T, buffer_gpu, sizeof(buffer_gpu),
                         buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))

    if jobz == 'N'
        return W
    elseif jobz == 'V'
        return W, A
    end
end

# Xsyevdx
function Xsyevdx!(jobz::Char, range::Char, uplo::Char, A::StridedCuMatrix{T};
                  vl::Real=0.0, vu::Real=Inf, il::Integer=1, iu::Integer=0) where {T <: BlasFloat}
    chkuplo(uplo)
    n = checksquare(A)
    R = real(T)
    (n ≥ 1) && (iu == 0) && (iu = n)
    (range == 'I') && !(1 ≤ il ≤ iu ≤ n) && throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu = $iu), which must be between 1 and n = $n"))
    (range == 'V') && (vl ≥ vu) && throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
    lda = max(1, stride(A, 2))
    info = CuVector{Cint}(undef, 1)
    W = CuVector{R}(undef, n)
    vl = Ref{R}(vl)
    vu = Ref{R}(vu)
    h_meig = Ref{Int64}(0)
    params = CuSolverParameters()

    function bufferSize()
        out_cpu = Ref{Csize_t}(0)
        out_gpu = Ref{Csize_t}(0)
        cusolverDnXsyevdx_bufferSize(dense_handle(), params, jobz, range, uplo, n,
                                     T, A, lda, vl, vu, il, iu, h_meig,
                                     R, W, T, out_gpu, out_cpu)
        out_gpu[], out_cpu[]
    end
    with_workspaces(bufferSize()...) do buffer_gpu, buffer_cpu
        cusolverDnXsyevdx(dense_handle(), params, jobz, range, uplo, n, T, A,
                          lda, vl, vu, il, iu, h_meig, R, W, T, buffer_gpu,
                          sizeof(buffer_gpu), buffer_cpu, sizeof(buffer_cpu), info)
    end

    flag = @allowscalar info[1]
    unsafe_free!(info)
    chkargsok(BlasInt(flag))

    if jobz == 'N'
        return W, h_meig[]
    elseif jobz == 'V'
        return W, A, h_meig[]
    end
end

# LAPACK
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        LinearAlgebra.LAPACK.sytrs!(uplo::Char, A::StridedCuMatrix{$elty}, p::CuVector{Int64}, B::StridedCuVecOrMat{$elty}) = CUSOLVER.sytrs!(uplo, A, p, B)
        LinearAlgebra.LAPACK.trtri!(uplo::Char, diag::Char, A::StridedCuMatrix{$elty}) = CUSOLVER.trtri!(uplo, diag, A)
    end
end

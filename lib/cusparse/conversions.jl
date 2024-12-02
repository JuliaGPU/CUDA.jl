export sort_csc, sort_csr, sort_coo

adjtrans_wrappers = ((identity, identity),
                     (M -> :(Transpose{T, <:$M}), M -> :(_sptranspose(parent($M)))),
                     (M -> :(Adjoint{T, <:$M}), M -> :(_spadjoint(parent($M)))))

# conversion routines between different sparse and dense storage formats

"""
    sparse(x::DenseCuMatrix; fmt=:csc)
    sparse(I::CuVector, J::CuVector, V::CuVector, [m, n]; fmt=:csc)

Return a sparse cuda matrix, with type determined by `fmt`.
Possible formats are :csc, :csr, :bsr, and :coo.
"""
function SparseArrays.sparse(x::DenseCuMatrix; fmt=:csc)
    if fmt == :csc
        return CuSparseMatrixCSC(x)
    elseif fmt == :csr
        return CuSparseMatrixCSR(x)
    elseif fmt == :bsr
        return CuSparseMatrixBSR(x)
    elseif fmt == :coo
        return CuSparseMatrixCOO(x)
    else
        error("Format :$fmt not available, use :csc, :csr, :bsr or :coo.")
    end
end

function SparseArrays.sparse(I::CuVector, J::CuVector, V::CuVector, args...; kwargs...)
    sparse(Cint.(I), Cint.(J), V, args...; kwargs...)
end

function SparseArrays.sparse(I::CuVector{Cint}, J::CuVector{Cint}, V::CuVector{Tv},
                             m=maximum(I), n=maximum(J);
                             fmt=:csc, combine=nothing) where Tv
    # we use COO as an intermediate format, as it's easy to construct from I/J/V.
    coo = CuSparseMatrixCOO{Tv}(I, J, V, (m, n))

    # find groups of values that correspond to the same position in the matrix.
    # if there's no duplicates, `groups` will just be a vector of ones.
    # otherwise, it will contain gaps of zeros that indicates duplicate values.
    coo = sort_coo(coo, 'R')
    groups = similar(I, Int)
    function find_groups(groups, I, J)
        i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        if i > length(groups)
            return
        end
        len = 0

        # check if we're at the start of a new group
        @inbounds groups[i] = i == 1 || I[i] != I[i-1] || J[i] != J[i-1]

        return
    end
    kernel = @cuda launch=false find_groups(groups, coo.rowInd, coo.colInd)
    config = launch_configuration(kernel.fun)
    threads = min(length(groups), config.threads)
    blocks = cld(length(groups), threads)
    kernel(groups, coo.rowInd, coo.colInd; threads, blocks)

    # if we got any group of more than one element, we need to combine them.
    # this may actually not be required, as some CUSPARSE functions can handle
    # duplicate entries, but it's not clear which ones do and which ones don't.
    # also, to ensure matrix display is correct, combine values eagerly.
    ngroups = mapreduce(!iszero, +, groups)
    if ngroups != length(groups)
        if combine === nothing
            combine = if Tv === Bool
                |
            else
                +
            end
        end

        # by scanning the mask of groups, we can find a mapping for old to new indices
        indices = accumulate(+, groups)

        I = similar(I, ngroups)
        J = similar(J, ngroups)
        V = similar(V, ngroups)

        # use one thread per (old) value, and if it's at the start of a group,
        # combine (if needed) all values and update the output vectors.
        function combine_groups(groups, indices, oldI, oldJ, oldV, newI, newJ, newV, combine)
            i = threadIdx().x + (blockIdx().x - 1) * blockDim().x
            if i > length(groups)
                return
            end

            # check if we're at the start of a group
            @inbounds if groups[i] != 0
                # get a destination index
                j = indices[i]

                # copy values
                newI[j] = oldI[i]
                newJ[j] = oldJ[i]
                val = oldV[i]
                while i < length(groups) && groups[i+1] == 0
                    i += 1
                    val = combine(val, oldV[i])
                end
                newV[j] = val
            end

            return
        end
        kernel = @cuda launch=false combine_groups(groups, indices, coo.rowInd, coo.colInd, coo.nzVal, I, J, V, combine)
        config = launch_configuration(kernel.fun)
        threads = min(length(groups), config.threads)
        blocks = cld(length(groups), threads)
        kernel(groups, indices, coo.rowInd, coo.colInd, coo.nzVal, I, J, V, combine; threads, blocks)
        synchronize()
        coo = CuSparseMatrixCOO{Tv}(I, J, V, (m, n))
    end

    if fmt == :coo
        return coo
    elseif fmt == :csc
        return CuSparseMatrixCSC(coo)
    elseif fmt == :csr
        return CuSparseMatrixCSR(coo)
    else
        error("Format :$fmt not available, use :csc, :csr, or :coo.")
    end
end

for (wrapa, unwrapa) in adjtrans_wrappers
    for SparseMatrixType in (:(CuSparseMatrixCSC{T}), :(CuSparseMatrixCSR{T}), :(CuSparseMatrixCOO{T}))
        TypeA = wrapa(SparseMatrixType)
        @eval SparseArrays.sparse(A::$TypeA) where {T} = $(unwrapa(:A))
    end
end

function sort_csc(A::CuSparseMatrixCSC{Tv,Ti}, index::SparseChar='O') where {Tv,Ti}

    m,n = size(A)
    perm = CuArray{Ti}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    sorted_colPtr = copy(A.colPtr)
    sorted_rowVal = copy(A.rowVal)
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseXcscsort_bufferSizeExt(handle(), m, n, nnz(A), A.colPtr, A.rowVal, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcscsort(handle(), m, n, nnz(A), descA, sorted_colPtr, sorted_rowVal, perm, buffer)
    end
    perm .+= one(Ti)
    sorted_nzVal = A.nzVal[perm]
    CUDA.unsafe_free!(perm)
    CuSparseMatrixCSC{Tv,Ti}(sorted_colPtr, sorted_rowVal, sorted_nzVal, size(A))
end

function sort_csr(A::CuSparseMatrixCSR{Tv,Ti}, index::SparseChar='O') where {Tv,Ti}

    m,n = size(A)
    perm = CuArray{Ti}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    descA = CuMatrixDescriptor('G', 'L', 'N', index)
    sorted_rowPtr = copy(A.rowPtr)
    sorted_colVal = copy(A.colVal)
    function bufferSize()
        out = Ref{Csize_t}()
        cusparseXcsrsort_bufferSizeExt(handle(), m, n, nnz(A), A.rowPtr, A.colVal, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseXcsrsort(handle(), m, n, nnz(A), descA, sorted_rowPtr, sorted_colVal, perm, buffer)
    end
    perm .+= one(Ti)
    sorted_nzVal = A.nzVal[perm]
    CUDA.unsafe_free!(perm)
    CuSparseMatrixCSR{Tv,Ti}(sorted_rowPtr, sorted_colVal, sorted_nzVal, size(A))
end

function sort_coo(A::CuSparseMatrixCOO{Tv,Ti}, type::SparseChar='R') where {Tv,Ti}

    type == 'R' || type == 'C' || throw(ArgumentError("type=$type was used and only type='R' and type='C' are supported."))

    m,n = size(A)
    perm = CuArray{Ti}(undef, nnz(A))
    cusparseCreateIdentityPermutation(handle(), nnz(A), perm)

    sorted_rowInd = copy(A.rowInd)
    sorted_colInd = copy(A.colInd)
    function bufferSize()
        # It seems that in some cases `out` is not updated
        # and we have the following error in the tests:
        # "Out of GPU memory trying to allocate 127.781 TiB".
        # We set 0 as default value to avoid it.
        out = Ref{Csize_t}(0)
        cusparseXcoosort_bufferSizeExt(handle(), m, n, nnz(A), A.rowInd, A.colInd, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        type == 'R' && cusparseXcoosortByRow(handle(), m, n, nnz(A), sorted_rowInd, sorted_colInd, perm, buffer)
        type == 'C' && cusparseXcoosortByColumn(handle(), m, n, nnz(A), sorted_rowInd, sorted_colInd, perm, buffer)
    end

    perm .+= one(Ti)
    sorted_nzVal = A.nzVal[perm]
    CUDA.unsafe_free!(perm)
    CuSparseMatrixCOO{Tv,Ti}(sorted_rowInd, sorted_colInd, sorted_nzVal, size(A))
end

for (bname, fname, pname, elty) in ((:cusparseSpruneCsr2csr_bufferSizeExt, :cusparseSpruneCsr2csrNnz, :cusparseSpruneCsr2csr, :Float32),
                                    (:cusparseDpruneCsr2csr_bufferSizeExt, :cusparseDpruneCsr2csrNnz, :cusparseDpruneCsr2csr, :Float64))
    @eval begin
        function prune(A::CuSparseMatrixCSR{$elty}, threshold::Number, index::SparseChar)
            m, n = size(A)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            rowPtrC = CuVector{Int32}(undef, m+1)
            local colValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), m, n, nnz(A), descA, nonzeros(A), A.rowPtr, A.colVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), rowPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), m, n, nnz(A), descA, nonzeros(A), A.rowPtr, A.colVal,
                       Ref{$elty}(threshold), descC, rowPtrC, nnzTotal, buffer)

                colValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), m, n, nnz(A), descA, nonzeros(A), A.rowPtr, A.colVal,
                       Ref{$elty}(threshold), descC, nzValC, rowPtrC, colValC, buffer)
            end
            return CuSparseMatrixCSR(rowPtrC, colValC, nzValC, (m, n))
        end

        function prune(A::CuSparseMatrixCSC{$elty}, threshold::Number, index::SparseChar)
            m, n = size(A)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            colPtrC = CuVector{Int32}(undef, n+1)
            local rowValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), n, m, nnz(A), descA, nonzeros(A), A.colPtr, A.rowVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), colPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), n, m, nnz(A), descA, nonzeros(A), A.colPtr, A.rowVal,
                       Ref{$elty}(threshold), descC, colPtrC, nnzTotal, buffer)

                rowValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), n, m, nnz(A), descA, nonzeros(A), A.colPtr, A.rowVal,
                       Ref{$elty}(threshold), descC, nzValC, colPtrC, rowValC, buffer)
            end
            return CuSparseMatrixCSC(colPtrC, rowValC, nzValC, (m, n))
        end
    end
end

## CSR to CSC

function CuSparseMatrixCSC{T}(S::Transpose{T, <:CuSparseMatrixCSR{T}}) where T
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, csr.nzVal, size(csr))
end

function CuSparseMatrixCSR{T}(S::Transpose{T, <:CuSparseMatrixCSC{T}}) where T
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, csc.nzVal, size(csc))
end

function CuSparseMatrixCSC{T}(S::Adjoint{T, <:CuSparseMatrixCSR{T}}) where {T <: Real}
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, csr.nzVal, size(csr))
end

function CuSparseMatrixCSR{T}(S::Adjoint{T, <:CuSparseMatrixCSC{T}}) where {T <: Real}
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, csc.nzVal, size(csc))
end

function CuSparseMatrixCSC{T}(S::Adjoint{T, <:CuSparseMatrixCSR{T}}) where {T <: Complex}
    csr = parent(S)
    return CuSparseMatrixCSC{T}(csr.rowPtr, csr.colVal, conj.(csr.nzVal), size(csr))
end

function CuSparseMatrixCSR{T}(S::Adjoint{T, <:CuSparseMatrixCSC{T}}) where {T <: Complex}
    csc = parent(S)
    return CuSparseMatrixCSR{T}(csc.colPtr, csc.rowVal, conj.(csc.nzVal), size(csc))
end

for SparseMatrixType in (:CuSparseMatrixCSC, :CuSparseMatrixCSR, :CuSparseMatrixCOO)
    @eval begin
        $SparseMatrixType(S::Diagonal{Tv, <:AbstractVector}) where {Tv} = $SparseMatrixType(cu(S))
        $SparseMatrixType(S::Diagonal{Tv, <:CuArray}) where Tv = $SparseMatrixType{Tv}(S)
        $SparseMatrixType{Tv}(S::Diagonal) where {Tv} = $SparseMatrixType{Tv, Cint}(S)
    end

    if SparseMatrixType == :CuSparseMatrixCOO
        @eval function $SparseMatrixType{Tv, Ti}(S::Diagonal) where {Tv, Ti}
            m = size(S, 1)
            return $SparseMatrixType{Tv, Ti}(CuVector(1:m), CuVector(1:m), convert(CuVector{Tv}, S.diag), (m, m))
        end
    else
        @eval function $SparseMatrixType{Tv, Ti}(S::Diagonal) where {Tv, Ti}
            m = size(S, 1)
            return $SparseMatrixType{Tv, Ti}(CuVector(1:(m+1)), CuVector(1:m), convert(CuVector{Tv}, S.diag), (m, m))
        end
    end
end

# by flipping rows and columns, we can use that to get CSC to CSR too
for elty in (:Float32, :Float64, :ComplexF32, :ComplexF64)
    @eval begin
        function CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty}; index::SparseChar='O', action::cusparseAction_t=CUSPARSE_ACTION_NUMERIC, algo::cusparseCsr2CscAlg_t=CUSPARSE_CSR2CSC_ALG1)
            m,n = size(csr)
            colPtr = CUDA.ones(Cint, n+1)
            rowVal = CUDA.zeros(Cint, nnz(csr))
            nzVal = CUDA.zeros($elty, nnz(csr))
            function bufferSize()
                out = Ref{Csize_t}(1)
                cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz(csr), nonzeros(csr),
                    csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                    $elty, action, index, algo, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                cusparseCsr2cscEx2(handle(), m, n, nnz(csr), nonzeros(csr),
                    csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                    $elty, action, index, algo, buffer)
            end
            CuSparseMatrixCSC(colPtr,rowVal,nzVal,size(csr))
        end

        function CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty}; index::SparseChar='O', action::cusparseAction_t=CUSPARSE_ACTION_NUMERIC, algo::cusparseCsr2CscAlg_t=CUSPARSE_CSR2CSC_ALG1)
            m,n    = size(csc)
            rowPtr = CUDA.ones(Cint,m+1)
            colVal = CUDA.zeros(Cint,nnz(csc))
            nzVal  = CUDA.zeros($elty,nnz(csc))
            function bufferSize()
                out = Ref{Csize_t}(1)
                cusparseCsr2cscEx2_bufferSize(handle(), n, m, nnz(csc), nonzeros(csc),
                    csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                    $elty, action, index, algo, out)
                return out[]
            end
            with_workspace(bufferSize) do buffer
                cusparseCsr2cscEx2(handle(), n, m, nnz(csc), nonzeros(csc),
                    csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                    $elty, action, index, algo, buffer)
            end
            CuSparseMatrixCSR(rowPtr,colVal,nzVal,size(csc))
        end
    end
end

# implement Float16 conversions using wider types
# TODO: Float16 is sometimes natively supported
for (elty, welty) in ((:Float16, :Float32),
                      (:ComplexF16, :ComplexF32))
    @eval begin
        function CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty}; index::SparseChar='O', action::cusparseAction_t=CUSPARSE_ACTION_NUMERIC, algo::cusparseCsr2CscAlg_t=CUSPARSE_CSR2CSC_ALG1)
            m,n = size(csr)
            colPtr = CUDA.ones(Cint, n+1)
            rowVal = CUDA.zeros(Cint, nnz(csr))
            nzVal = CUDA.zeros($elty, nnz(csr))
            if $elty == Float16 #broken for ComplexF16?
                function bufferSize()
                    out = Ref{Csize_t}(1)
                    cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz(csr), nonzeros(csr),
                        csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                        $elty, action, index, algo, out)
                    return out[]
                end
                with_workspace(bufferSize) do buffer
                    cusparseCsr2cscEx2(handle(), m, n, nnz(csr), nonzeros(csr),
                        csr.rowPtr, csr.colVal, nzVal, colPtr, rowVal,
                        $elty, action, index, algo, buffer)
                end
                return CuSparseMatrixCSC(colPtr,rowVal,nzVal,size(csr))
            else
                wide_csr = CuSparseMatrixCSR(csr.rowPtr, csr.colVal, convert(CuVector{$welty}, nonzeros(csr)), size(csr))
                wide_csc = CuSparseMatrixCSC(wide_csr)
                return CuSparseMatrixCSC(wide_csc.colPtr, wide_csc.rowVal, convert(CuVector{$elty}, nonzeros(wide_csc)), size(wide_csc))
            end
        end

        function CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty}; index::SparseChar='O', action::cusparseAction_t=CUSPARSE_ACTION_NUMERIC, algo::cusparseCsr2CscAlg_t=CUSPARSE_CSR2CSC_ALG1)
            m,n    = size(csc)
            rowPtr = CUDA.ones(Cint,m+1)
            colVal = CUDA.zeros(Cint,nnz(csc))
            nzVal  = CUDA.zeros($elty,nnz(csc))
            if $elty == Float16 #broken for ComplexF16?
                function bufferSize()
                    out = Ref{Csize_t}(1)
                    cusparseCsr2cscEx2_bufferSize(handle(), n, m, nnz(csc), nonzeros(csc),
                        csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                        $elty, action, index, algo, out)
                    return out[]
                end
                with_workspace(bufferSize) do buffer
                    cusparseCsr2cscEx2(handle(), n, m, nnz(csc), nonzeros(csc),
                        csc.colPtr, rowvals(csc), nzVal, rowPtr, colVal,
                        $elty, action, index, algo, buffer)
                end
                return CuSparseMatrixCSR(rowPtr,colVal,nzVal,size(csc))
            else
                wide_csc = CuSparseMatrixCSC(csc.colPtr, csc.rowVal, convert(CuVector{$welty}, nonzeros(csc)), size(csc))
                wide_csr = CuSparseMatrixCSR(wide_csc)
                return CuSparseMatrixCSR(wide_csr.rowPtr, wide_csr.colVal, convert(CuVector{$elty}, nonzeros(wide_csr)), size(wide_csr))
            end
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(csc::CuSparseMatrixCSC{$elty})
            csc_compat = CuSparseMatrixCSC(
                csc.colPtr,
                csc.rowVal,
                reinterpret($felty, csc.nzVal),
                size(csc)
            )
            csr_compat = CuSparseMatrixCSR(csc_compat)
            CuSparseMatrixCSR(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixCSC{$elty}(csr::CuSparseMatrixCSR{$elty})
            csr_compat = CuSparseMatrixCSR(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            csc_compat = CuSparseMatrixCSC(csr_compat)
            CuSparseMatrixCSC(
                csc_compat.colPtr,
                csc_compat.rowVal,
                reinterpret($elty, csc_compat.nzVal),
                size(csc_compat)
            )
        end
    end
end

## CuSparseVector to CuVector
CuVector(x::CuSparseVector{T}) where {T} = CuVector{T}(x)

function CuVector{T}(sv::CuSparseVector{T}) where {T}
    n = length(sv)
    dv = CUDA.zeros(T, n)
    scatter!(dv, sv, 'O')
end

## CSR to BSR and vice-versa

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty}, blockDim::Integer;
                                          dir::SparseChar='R', index::SparseChar='O',
                                          indc::SparseChar='O')
            m,n = size(csr)
            nnz_ref = Ref{Cint}(1)
            mb = cld(m, blockDim)
            nb = cld(n, blockDim)
            bsrRowPtr = CUDA.ones(Cint,mb + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csr.rowPtr,
                                csr.colVal, blockDim, cudescc, bsrRowPtr, nnz_ref)
            (nnz_ref[] > mb * nb) && error("The number of nonzero blocks of the BSR matrix is incorrect.")
            bsrNzVal = CUDA.zeros($elty, nnz_ref[] * blockDim * blockDim )
            bsrColInd = CUDA.zeros(Cint, nnz_ref[])
            $fname(handle(), dir, m, n,
                   cudesca, nonzeros(csr), csr.rowPtr, csr.colVal,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            CuSparseMatrixBSR{$elty}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, nnz_ref[])
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty};
                                          index::SparseChar='O', indc::SparseChar='O')
            m,n = size(bsr)
            mb = cld(m, bsr.blockDim)
            nb = cld(n, bsr.blockDim)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            csrRowPtr = CUDA.ones(Cint, m + 1)
            csrColInd = CUDA.zeros(Cint, nnz(bsr))
            csrNzVal  = CUDA.zeros($elty, nnz(bsr))
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, nonzeros(bsr), bsr.rowPtr, bsr.colVal,
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            # XXX: the size here may not match the expected size, when the matrix dimension
            #      is not a multiple of the block dimension!
            CuSparseMatrixCSR(csrRowPtr, csrColInd, csrNzVal, (mb*bsr.blockDim, nb*bsr.blockDim))
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty})
            bsr_compat = CuSparseMatrixBSR(
                bsr.rowPtr,
                bsr.colVal,
                reinterpret($felty, bsr.nzVal),
                bsr.blockDim,
                bsr.dir,
                bsr.nnzb,
                size(bsr)
            )
            csr_compat = CuSparseMatrixCSR(bsr_compat)
            CuSparseMatrixCSR(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty}, blockDim)
            csr_compat = CuSparseMatrixCSR(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            bsr_compat = CuSparseMatrixBSR(csr_compat, blockDim)
            CuSparseMatrixBSR(
                bsr_compat.rowPtr,
                bsr_compat.colVal,
                reinterpret($elty, bsr_compat.nzVal),
                bsr_compat.blockDim,
                bsr_compat.dir,
                bsr_compat.nnzb,
                size(bsr_compat)
            )
        end
    end
end

## CSR to COO and vice-versa

function CuSparseMatrixCSR{Tv}(coo::CuSparseMatrixCOO{Tv}; index::SparseChar='O') where {Tv}
    m,n = size(coo)
    coo = sort_coo(coo, 'R')
    csrRowPtr = CuVector{Cint}(undef, m+1)
    cusparseXcoo2csr(handle(), coo.rowInd, nnz(coo), m, csrRowPtr, index)
    CuSparseMatrixCSR{Tv}(csrRowPtr, coo.colInd, nonzeros(coo), size(coo))
end

function CuSparseMatrixCOO{Tv}(csr::CuSparseMatrixCSR{Tv}; index::SparseChar='O') where {Tv}
    m,n = size(csr)
    cooRowInd = CuVector{Cint}(undef, nnz(csr))
    cusparseXcsr2coo(handle(), csr.rowPtr, nnz(csr), m, cooRowInd, index)
    CuSparseMatrixCOO{Tv}(cooRowInd, csr.colVal, nonzeros(csr), size(csr))
end

### CSC to COO and viceversa

function CuSparseMatrixCSC{Tv}(coo::CuSparseMatrixCOO{Tv}; index::SparseChar='O') where {Tv}
    m,n = size(coo)
    coo = sort_coo(coo, 'C')
    cscColPtr = CuVector{Cint}(undef, n+1)
    cusparseXcoo2csr(handle(), coo.colInd, nnz(coo), n, cscColPtr, index)
    CuSparseMatrixCSC{Tv}(cscColPtr, coo.rowInd, nonzeros(coo), size(coo))
end

function CuSparseMatrixCOO{Tv}(csc::CuSparseMatrixCSC{Tv}; index::SparseChar='O') where {Tv}
    m,n = size(csc)
    cooColInd = CuVector{Cint}(undef, nnz(csc))
    cusparseXcsr2coo(handle(), csc.colPtr, nnz(csc), n, cooColInd, index)
    coo = CuSparseMatrixCOO{Tv}(csc.rowVal, cooColInd, nonzeros(csc), size(csc))
    coo = sort_coo(coo, 'R')
end

### BSR to COO and vice-versa

CuSparseMatrixBSR(coo::CuSparseMatrixCOO, blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(coo), blockdim) # no direct conversion
CuSparseMatrixCOO(bsr::CuSparseMatrixBSR) = CuSparseMatrixCOO(CuSparseMatrixCSR(bsr)) # no direct conversion

### BSR to CSC and vice-versa

CuSparseMatrixBSR(csc::CuSparseMatrixCSC, blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(csc), blockdim) # no direct conversion
CuSparseMatrixCSC(bsr::CuSparseMatrixBSR) = CuSparseMatrixCSC(CuSparseMatrixCSR(bsr)) # no direct conversion

## sparse to dense, and vice-versa
function CUDA.CuMatrix{T}(csr::CuSparseMatrixCSR{T}; index::SparseChar='O') where {T}
    denseA = sparsetodense(csr, index)
    return denseA
end

function CUDA.CuMatrix{T}(csc::CuSparseMatrixCSC{T}; index::SparseChar='O') where {T}
    denseA = sparsetodense(csc, index)
    return denseA
end

Base.copyto!(dest::Matrix{T}, src::AbstractCuSparseMatrix{T}) where T = copyto!(dest, CuMatrix{T}(src))

function CuSparseMatrixCSR(A::CuMatrix{T}; index::SparseChar='O', sorted::Bool=false) where {T}
    csr = densetosparse(A, :csr, index)
    csr = sorted ? sort_csr(csr, index) : csr
    return csr
end

function CuSparseMatrixCSC(A::CuMatrix{T}; index::SparseChar='O', sorted::Bool=false) where {T}
    csc = densetosparse(A, :csc, index)
    csc = sorted ? sort_csc(csc, index) : csc
    return csc
end

function CUDA.CuMatrix{T}(bsr::CuSparseMatrixBSR{T}; index::SparseChar='O',
                          indc::SparseChar='O') where {T}
    CuMatrix{T}(CuSparseMatrixCSR{T}(bsr; index, indc))
end

function CuSparseMatrixBSR(A::CuMatrix, blockDim::Integer=gcd(size(A)...); index::SparseChar='O')
    m,n = size(A)
    # csr.colVal should be sorted if we want to use "csr2bsr" routines.
    csr = CuSparseMatrixCSR(A; index, sorted=true)
    CuSparseMatrixBSR(csr, blockDim)
end

function CUDA.CuMatrix{T}(coo::CuSparseMatrixCOO{T}; index::SparseChar='O') where {T}
    sparsetodense(coo, index)
end

function CuSparseMatrixCOO(A::CuMatrix{T}; index::SparseChar='O') where {T}
    densetosparse(A, :coo, index)
end

## CuSparseVector to CuSparseMatrices and vice-versa
function CuSparseVector(A::CuSparseMatrixCSC{T}) where T
    m, n = size(A)
    (n == 1) || error("A doesn't have one column and can't be converted to a CuSparseVector.")
    CuSparseVector{T}(A.rowVal, A.nzVal, m)
end

# no direct conversion
function CuSparseVector(A::CuSparseMatrixCSR{T}) where T
    m, n = size(A)
    (n == 1) || error("A doesn't have one column and can't be converted to a CuSparseVector.")
    B = CuSparseMatrixCSC{T}(A)
    CuSparseVector(B)
end

function CuSparseVector(A::CuSparseMatrixCOO{T}) where T
    m, n = size(A)
    (n == 1) || error("A doesn't have one column and can't be converted to a CuSparseVector.")
    CuSparseVector{T}(A.rowInd, A.nzVal, m)
end

function CuSparseMatrixCSC(x::CuSparseVector{T}) where T
    n = length(x)
    colPtr = CuVector{Int32}([1; nnz(x)+1])
    CuSparseMatrixCSC{T}(colPtr, x.iPtr, nonzeros(x), (n,1))
end

# no direct conversion
function CuSparseMatrixCSR(x::CuSparseVector{T}) where T
    A = CuSparseMatrixCSC(x)
    CuSparseMatrixCSR{T}(A)
end

function CuSparseMatrixCOO(x::CuSparseVector{T}) where T
    n = length(x)
    nnzx = nnz(x)
    colInd = CuVector{Int32}(undef, nnzx)
    fill!(colInd, one(Int32))
    CuSparseMatrixCOO{T}(x.iPtr, colInd, nonzeros(x), (n,1), nnzx)
end

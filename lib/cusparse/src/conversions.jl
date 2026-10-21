# Qualified vendor helpers. Ordinary format constructors delegate to GPUArrays.
export sort_csc, sort_csr, sort_coo

# Canonical inputs are already sorted. Column-sorted COO is not a canonical array.
sort_csc(A::CuSparseMatrixCSC,index::SparseChar='O') = (index == 'O' || throw(ArgumentError("canonical arrays use one-based indices")); copy(A))
sort_csr(A::CuSparseMatrixCSR,index::SparseChar='O') = (index == 'O' || throw(ArgumentError("canonical arrays use one-based indices")); copy(A))
function sort_coo(A::CuSparseMatrixCOO,order::SparseChar='R')
    order == 'R' || throw(ArgumentError("column-sorted COO is not a canonical GPUArrays array; use CSC or keep sorted coordinates as separate buffers"))
    return copy(A)
end

function check_cint_indices(op, sizes::Integer...)
    for sz in sizes
        sz <= typemax(Cint) || throw(ArgumentError(
            "$op uses 32-bit indices in cuSPARSE, which cannot represent a size of $sz"))
    end
    return
end

cint_indices(x::CuVector) = convert(CuVector{Cint}, x)


for (bname, fname, pname, elty) in ((:cusparseSpruneCsr2csr_bufferSizeExt, :cusparseSpruneCsr2csrNnz, :cusparseSpruneCsr2csr, :Float32),
                                    (:cusparseDpruneCsr2csr_bufferSizeExt, :cusparseDpruneCsr2csrNnz, :cusparseDpruneCsr2csr, :Float64))
    @eval begin
        function prune(A::CuSparseMatrixCSR{$elty, Ti}, threshold::Number, index::SparseChar) where {Ti}
            index=='O' || throw(ArgumentError("canonical arrays use one-based indices"))
            m, n = size(A)
            check_cint_indices("prune", m, n, nnz(A) + 1)
            rowPtr = cint_indices(A.rowPtr)
            colVal = cint_indices(A.colVal)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            rowPtrC = CuVector{Int32}(undef, m+1)
            local colValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), rowPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, rowPtrC, nnzTotal, buffer)

                colValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), m, n, nnz(A), descA, nonzeros(A), rowPtr, colVal,
                       Ref{$elty}(threshold), descC, nzValC, rowPtrC, colValC, buffer)
            end
            return CuSparseMatrixCSR(GPUArrays.convert_index_buffer(eltype(A.rowPtr),rowPtrC), GPUArrays.convert_index_buffer(Ti,colValC), nzValC, (m,n))
        end

        function prune(A::CuSparseMatrixCSC{$elty, Ti}, threshold::Number, index::SparseChar) where {Ti}
            m, n = size(A)
            check_cint_indices("prune", m, n, nnz(A) + 1)
            colPtr = cint_indices(A.colPtr)
            rowVal = cint_indices(A.rowVal)
            descA = CuMatrixDescriptor('G', 'L', 'N', index)
            descC = CuMatrixDescriptor('G', 'L', 'N', index)
            colPtrC = CuVector{Int32}(undef, n+1)
            local rowValC, nzValC

            function bufferSize()
                out = Ref{Csize_t}()
                $bname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, CuPtr{$elty}(CU_NULL), colPtrC, CuPtr{Int32}(CU_NULL), out)
                return out[]
            end

            with_workspace(bufferSize) do buffer
                nnzTotal = Ref{Cint}()
                $fname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, colPtrC, nnzTotal, buffer)

                rowValC = CuVector{Int32}(undef, nnzTotal[])
                nzValC  = CuVector{$elty}(undef, nnzTotal[])

                $pname(handle(), n, m, nnz(A), descA, nonzeros(A), colPtr, rowVal,
                       Ref{$elty}(threshold), descC, nzValC, colPtrC, rowValC, buffer)
            end
            return CuSparseMatrixCSC(GPUArrays.convert_index_buffer(eltype(A.colPtr),colPtrC), GPUArrays.convert_index_buffer(Ti,rowValC), nzValC, (m,n))
        end
    end
end


function _csr2cscEx2!(m, n, nnz_, csrVal::CuVector{T}, csrRowPtr, csrColInd,
                      cscVal::CuVector{T}, cscColPtr::CuVector{Cint}, cscRowInd::CuVector{Cint},
                      action, index, algo) where T
    csrRowPtr = cint_indices(csrRowPtr)
    csrColInd = cint_indices(csrColInd)
    if iszero(nnz_) || iszero(m) || iszero(n)
        cscColPtr .= (index == 'O' ? one(Cint) : zero(Cint))
        return
    end

    buggy = index == 'O' && v"12.0" <= version() < v"12.1"
    if buggy
        csrRowPtr = csrRowPtr .- one(Cint)
        csrColInd = csrColInd .- one(Cint)
        effidx = 'Z'
    else
        effidx = index
    end
    function bufferSize()
        out = Ref{Csize_t}(1)
        cusparseCsr2cscEx2_bufferSize(handle(), m, n, nnz_, csrVal,
            csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
            T, action, effidx, algo, out)
        return out[]
    end
    with_workspace(bufferSize) do buffer
        cusparseCsr2cscEx2(handle(), m, n, nnz_, csrVal,
            csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
            T, action, effidx, algo, buffer)
    end
    if buggy
        cscColPtr .+= one(Cint)
        cscRowInd .+= one(Cint)
    end
    return
end


## CSR to BSR and vice-versa

for (fname,elty) in ((:cusparseScsr2bsr, :Float32),
                     (:cusparseDcsr2bsr, :Float64),
                     (:cusparseCcsr2bsr, :ComplexF32),
                     (:cusparseZcsr2bsr, :ComplexF64))
    @eval begin
        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Ti}, blockDim::Integer;
                                          dir::SparseChar='R', index::SparseChar='O',
                                          indc::SparseChar='O') where {Ti}
            index==indc=='O' || throw(ArgumentError("canonical arrays use one-based indices"))
            blockDim>0 || throw(ArgumentError("block dimension must be positive"))
            m,n = size(csr)
            mb = cld(m, blockDim)
            nb = cld(n, blockDim)
            # cusparseXcsr2bsrNnz rejects zero-dim inputs with CUSPARSE_STATUS_INVALID_VALUE;
            # the result is unambiguously empty, so construct it directly.
            if iszero(m) || iszero(n)
                bsrRowPtr = CuArray{Ti}(undef, mb + 1)
                bsrRowPtr .= (indc == 'O' ? one(Ti) : zero(Ti))
                bsrColInd = CuArray{Ti}(undef, 0)
                bsrNzVal  = CuArray{$elty}(undef, 0)
                return CuSparseMatrixBSR{$elty, Ti}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, 0)
            end
            check_cint_indices("CSR to BSR conversion", m, n, nnz(csr) + 1)
            csrRowPtr = cint_indices(csr.rowPtr)
            csrColInd = cint_indices(csr.colVal)
            nnz_ref = Ref{Cint}(1)
            bsrRowPtr = CUDACore.zeros(Cint, mb + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            cusparseXcsr2bsrNnz(handle(), dir, m, n, cudesca, csrRowPtr,
                                csrColInd, blockDim, cudescc, bsrRowPtr, nnz_ref)
            (nnz_ref[] > mb * nb) && error("The number of nonzero blocks of the BSR matrix is incorrect.")
            bsrNzVal = CUDACore.zeros($elty, nnz_ref[] * blockDim * blockDim )
            bsrColInd = CUDACore.zeros(Cint, nnz_ref[])
            $fname(handle(), dir, m, n,
                   cudesca, nonzeros(csr), csrRowPtr, csrColInd,
                   blockDim, cudescc, bsrNzVal, bsrRowPtr,
                   bsrColInd)
            CuSparseMatrixBSR{$elty, Ti}(bsrRowPtr, bsrColInd, bsrNzVal, size(csr), blockDim, dir, nnz_ref[])
        end
    end
end

for (fname,elty) in ((:cusparseSbsr2csr, :Float32),
                     (:cusparseDbsr2csr, :Float64),
                     (:cusparseCbsr2csr, :ComplexF32),
                     (:cusparseZbsr2csr, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Ti};
                                          index::SparseChar='O', indc::SparseChar='O') where {Ti}
            index==indc=='O' || throw(ArgumentError("canonical arrays use one-based indices"))
            m,n = size(bsr)
            if iszero(bsr.nnzb)
                return CuSparseMatrixCSR(fill!(CuVector{Ti}(undef,m+1),one(Ti)),
                    CuVector{Ti}(undef,0),CuVector{$elty}(undef,0),size(bsr))
            end
            mb = cld(m, bsr.blockDim)
            nb = cld(n, bsr.blockDim)
            check_cint_indices("BSR to CSR conversion", widemul(mb, bsr.blockDim),
                               widemul(nb, bsr.blockDim), widemul(bsr.nnzb, widemul(bsr.blockDim, bsr.blockDim)) + 1)
            cudesca = CuMatrixDescriptor('G', 'L', 'N', index)
            cudescc = CuMatrixDescriptor('G', 'L', 'N', indc)
            padded_m = Base.checked_mul(mb, Int(bsr.blockDim))
            padded_n = Base.checked_mul(nb, Int(bsr.blockDim))
            csrRowPtr = CuArray{Cint}(undef, Base.checked_add(padded_m, 1))
            # cusparse*bsr2csr does not initialize csrRowPtr when the block grid
            # is empty; pre-fill with the empty-matrix sentinel.
            if iszero(mb) || iszero(nb)
                csrRowPtr .= (indc == 'O' ? one(Cint) : zero(Cint))
            end
            csrColInd = CUDACore.zeros(Cint, nnz(bsr))
            csrNzVal  = CUDACore.zeros($elty, nnz(bsr))
            $fname(handle(), bsr.dir, mb, nb,
                   cudesca, nonzeros(bsr), cint_indices(bsr.rowPtr), cint_indices(bsr.colVal),
                   bsr.blockDim, cudescc, csrNzVal, csrRowPtr,
                   csrColInd)
            result = canonical_csr(csrRowPtr, csrColInd, csrNzVal, (padded_m,padded_n), Ti, Ti)
            size(result) == size(bsr) ? result : result[1:m,1:n]
        end
    end
end

# implement Int conversions using reinterpreted Float
for (elty, felty) in ((:Int16, :Float16),
                      (:Int32, :Float32),
                      (:Int64, :Float64),
                      (:Int128, :ComplexF64))
    @eval begin
        function CuSparseMatrixCSR{$elty}(bsr::CuSparseMatrixBSR{$elty, Ti};
                                          index::SparseChar='O', indc::SparseChar='O') where {Ti}
            bsr_compat = CuSparseMatrixBSR{$felty, Ti}(
                bsr.rowPtr,
                bsr.colVal,
                reinterpret($felty, bsr.nzVal),
                size(bsr),
                bsr.blockDim,
                bsr.dir,
                bsr.nnzb
            )
            csr_compat = CuSparseMatrixCSR{$felty}(bsr_compat; index, indc)
            CuSparseMatrixCSR{$elty, Ti}(
                csr_compat.rowPtr,
                csr_compat.colVal,
                reinterpret($elty, csr_compat.nzVal),
                size(csr_compat)
            )
        end

        function CuSparseMatrixBSR{$elty}(csr::CuSparseMatrixCSR{$elty, Ti}, blockDim) where {Ti}
            csr_compat = CuSparseMatrixCSR{$felty, Ti}(
                csr.rowPtr,
                csr.colVal,
                reinterpret($felty, csr.nzVal),
                size(csr)
            )
            bsr_compat = CuSparseMatrixBSR(csr_compat, blockDim)
            CuSparseMatrixBSR{$elty, Ti}(
                bsr_compat.rowPtr,
                bsr_compat.colVal,
                reinterpret($elty, bsr_compat.nzVal),
                size(bsr_compat),
                bsr_compat.blockDim,
                bsr_compat.dir,
                bsr_compat.nnzb
            )
        end
    end
end



# Normalize owned vendor output before it becomes a canonical array. Pointer width is
# separate from coordinate width and is chosen by the operation, not by the library ABI.
function canonical_csr(ptr,ind,values,dims,::Type{Ti}=eltype(ind),::Type{Tp}=eltype(ptr)) where {Ti,Tp}
    rows = GPUArrays.expand_ptr(ptr,length(values),Ti)
    cols = GPUArrays.convert_index_buffer(Ti,ind)
    return GPUArrays.assemble_matrix(rows,cols,values,dims...,GPUArrays.default_combine(eltype(values)),:csr;
                                     pointer_type=Tp)
end

CuSparseMatrixBSR(A::CuSparseMatrixCOO,b) = CuSparseMatrixBSR(CuSparseMatrixCSR(A),b)
CuSparseMatrixBSR{T,Ti}(A::CuCanonicalMatrix,b::Integer;kwargs...) where {T,Ti} =
    CuSparseMatrixBSR(CuSparseMatrixCSR{T,Ti}(A),b;kwargs...)
CuSparseMatrixBSR{T}(A::Union{CuSparseMatrixCSC{<:Any,Ti},CuSparseMatrixCOO{<:Any,Ti}},b::Integer;kwargs...) where {T,Ti} =
    CuSparseMatrixBSR{T,Ti}(A,b;kwargs...)
CuSparseMatrixCOO(A::CuSparseMatrixBSR) = CuSparseMatrixCOO(CuSparseMatrixCSR(A))
CuSparseMatrixBSR(A::CuSparseMatrixCSC,b) = CuSparseMatrixBSR(CuSparseMatrixCSR(A),b)
CuSparseMatrixCSC(A::CuSparseMatrixBSR) = CuSparseMatrixCSC(CuSparseMatrixCSR(A))
CuSparseMatrixCSR(A::CuSparseMatrixBSR{T};kwargs...) where {T} = CuSparseMatrixCSR{T}(A;kwargs...)
CuSparseMatrixBSR(A::CuSparseMatrixCSR{T},b;kwargs...) where {T} = CuSparseMatrixBSR{T}(A,b;kwargs...)
CuSparseMatrixCSR{T,Ti}(A::CuSparseMatrixBSR) where {T,Ti} = GPUArrays.with_eltypes(CuSparseMatrixCSR(A),T,Ti)
CuSparseMatrixCSC{T,Ti}(A::CuSparseMatrixBSR) where {T,Ti} = GPUArrays.with_eltypes(CuSparseMatrixCSC(A),T,Ti)
CuSparseMatrixCOO{T,Ti}(A::CuSparseMatrixBSR) where {T,Ti} = GPUArrays.with_eltypes(CuSparseMatrixCOO(A),T,Ti)
function CuSparseMatrixBSR(A::CuMatrix,b::Integer=max(gcd(size(A)...),1);index::SparseChar='O')
    CuSparseMatrixBSR(CuSparseMatrixCSR(A),b;index)
end
function SparseArrays.sparse(A::CuMatrix;fmt::Symbol=:csc)
    fmt===:bsr ? CuSparseMatrixBSR(A) : GPUArrays.sparse_format(fmt){eltype(A),Cint}(A)
end
CUDACore.CuMatrix{T}(A::CuSparseMatrixBSR;index::SparseChar='O',indc::SparseChar='O') where {T} =
    CuMatrix{T}(CuSparseMatrixCSR{T}(A;index,indc))

# Compatibility keywords no longer permit noncanonical outputs.
for (alias,generic) in ((:CuSparseMatrixCSR,:GPUSparseMatrixCSR),(:CuSparseMatrixCSC,:GPUSparseMatrixCSC))
    @eval function $alias(A::CuMatrix{T}; index::SparseChar='O', sorted::Bool=true) where {T}
        index == 'O' || throw(ArgumentError("canonical arrays use one-based indices"))
        sorted || Base.depwarn("sorted=false no longer permits noncanonical sparse arrays",:sparse )
        return $generic{T,Cint}(A)
    end
end
CuSparseMatrixCOO(A::CuMatrix{T};index::SparseChar='O') where {T} =
    (index == 'O' || throw(ArgumentError("canonical arrays use one-based indices")); GPUSparseMatrixCOO{T,Cint}(A))

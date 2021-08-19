mutable struct CuSparseDeviceVector{Tv} <: AbstractCuSparseVector{Tv}
    iPtr::CuDeviceVector{Cint, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2,Int}
    nnz::Cint
end

Base.length(g::CuSparseDeviceVector) = prod(g.dims)
Base.size(g::CuSparseDeviceVector) = g.dims
Base.ndims(g::CuSparseDeviceVector) = 1

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseVector{Tv}) where Tv
    CuSparseDeviceVector(cudaconvert(x.iPtr), cudaconvert(x.nzVal), x.dims, x.nnz)
end

mutable struct CuSparseDeviceMatrixCSC{Tv} <: AbstractCuSparseMatrix{Tv}
    colPtr::CuDeviceVector{Cint, AS.Global}
    rowVal::CuDeviceVector{Cint, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2,Int}
    nnz::Cint
end

Base.length(g::CuSparseDeviceMatrixCSC) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSC) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSC) = 2

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseDeviceMatrixCSC{Tv}) where Tv
    CuSparseDeviceMatrixCSR(
        cudaconvert(x.colPtr),
        cudaconvert(x.rowVal),
        cudaconvert(x.nzVal),
        x.dims, x.nnz
    )
end

struct CuSparseDeviceMatrixCSR{Tv} <: AbstractCuSparseMatrix{Tv}
    rowPtr::CuDeviceVector{Cint, AS.Global}
    colVal::CuDeviceVector{Cint, AS.Global}
    nzVal::CuDeviceVector{Tv, AS.Global}
    dims::NTuple{2, Int}
    nnz::Cint
end

Base.length(g::CuSparseDeviceMatrixCSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCSR) = 2

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseMatrixCSR{Tv}) where Tv
    CuSparseDeviceMatrixCSR(cudaconvert(x.rowPtr), cudaconvert(x.colVal), cudaconvert(x.nzVal), x.dims, x.nnz)
end

mutable struct CuSparseDeviceMatrixBSR{Tv} <: AbstractCuSparseMatrix{Tv}
    rowPtr::CuVector{Cint}
    colVal::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    blockDim::Cint
    dir::SparseChar
    nnz::Cint
end

Base.length(g::CuSparseDeviceMatrixBSR) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixBSR) = g.dims
Base.ndims(g::CuSparseDeviceMatrixBSR) = 2

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseDeviceMatrixBSR{Tv}) where Tv
    CuSparseDeviceMatrixBSR(
        cudaconvert(x.rowPtr),
        cudaconvert(x.colVal),
        cudaconvert(x.nzVal),
        x.dims, x.blockDim,
        x.dir, x.nnz
    )
end

struct CuSparseDeviceMatrixCOO{Tv} <: AbstractCuSparseMatrix{Tv}
    rowInd::CuVector{Cint}
    colInd::CuVector{Cint}
    nzVal::CuVector{Tv}
    dims::NTuple{2,Int}
    nnz::Cint
end

Base.length(g::CuSparseDeviceMatrixCOO) = prod(g.dims)
Base.size(g::CuSparseDeviceMatrixCOO) = g.dims
Base.ndims(g::CuSparseDeviceMatrixCOO) = 2

function Adapt.adapt_structure(to::CUDA.Adaptor, x::CuSparseDeviceMatrixCOO{Tv}) where Tv
    CuSparseDeviceMatrixCOO(
        cudaconvert(x.rowInd),
        cudaconvert(x.colInd),
        cudaconvert(x.nzVal),
        x.dims, x.nnz
    )
end

function Base.show(io::IO, ::MIME"text/plain", A::CuSparseDeviceMatrixCSR)
    println(io, "$(length(A))-element device sparse matrix CSR at:")
    println(io, "  rowPtr $(pointer(A.rowPtr))")
    println(io, "  colVal $(pointer(A.colVal))")
    print(io, "  nzVal $(pointer(A.nzVal))")
end

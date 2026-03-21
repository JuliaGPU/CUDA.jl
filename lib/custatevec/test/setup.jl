using Test, LinearAlgebra
using CUDA
using cuStateVec
import cuStateVec: CuStateVec, applyMatrix!, applyMatrixBatched!, applyPauliExp!,
    applyGeneralizedPermutationMatrix!, expectation, expectationsOnPauliBasis, sample,
    testMatrixType, Pauli, PauliX, PauliY, PauliZ, PauliI, measureOnZBasis!,
    swapIndexBits!, abs2SumOnZBasis, collapseOnZBasis!, batchMeasure!,
    batchMeasureWithOffset!, abs2SumArray, collapseByBitString!, abs2SumArrayBatched,
    collapseByBitStringBatched!, accessorSet!, accessorGet, CuStateVecAccessor

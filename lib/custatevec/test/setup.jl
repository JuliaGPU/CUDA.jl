using Test, LinearAlgebra
using CUDACore
using cuStateVec
import cuStateVec: CuStateVec, applyMatrix!, applyMatrixBatched!, applyPauliExp!,
    applyGeneralizedPermutationMatrix!, expectation, expectationBatched,
    expectationsOnPauliBasis, sample,
    testMatrixType, Pauli, PauliX, PauliY, PauliZ, PauliI, measureOnZBasis!,
    swapIndexBits!, abs2SumOnZBasis, collapseOnZBasis!, batchMeasure!,
    measureBatched!, batchMeasureWithOffset!, abs2SumArray, collapseByBitString!,
    abs2SumArrayBatched, collapseByBitStringBatched!, accessorSet!, accessorGet,
    CuStateVecAccessor

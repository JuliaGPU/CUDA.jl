using Test, LinearAlgebra, Random
using CUDACore
using cuStateVec
import cuStateVec: CuStateVec, applyMatrix!, applyMatrixBatched!, applyPauliRotation!,
    applyGeneralizedPermutationMatrix!, computeExpectation, computeExpectationBatched,
    computeExpectationsOnPauliBasis, sample,
    testMatrixType, Pauli, PauliX, PauliY, PauliZ, PauliI, measureOnZBasis!,
    swapIndexBits!, abs2SumOnZBasis, collapseOnZBasis!, batchMeasure!,
    measureBatched!, batchMeasureWithOffset!, abs2SumArray, collapseByBitString!,
    abs2SumArrayBatched, collapseByBitStringBatched!, accessorSet!, accessorGet,
    CuStateVecAccessor

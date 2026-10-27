using CUDATools, cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND
using cuDNN, cuTENSOR, cuTensorNet, cuStateVec
# trigger CUDACore's extensions
using ChainRulesCore, KernelAbstractions, SpecialFunctions, StaticArrays

# captured variables that get boxed (`Core.Box`) lose their type
if isdefined(Test, :detect_closure_boxes)
    exts = [Base.get_extension(CUDACore, ext) for ext in (:ChainRulesCoreExt,
            :KernelAbstractionsExt, :SpecialFunctionsExt, :StaticArraysExt)]
    @test Test.detect_closure_boxes(CUDA, CUDACore, CUDATools, CUPTI, NVML,
                                    cuBLAS, cuSPARSE, cuSOLVER, cuFFT, cuRAND,
                                    cuDNN, cuTENSOR, cuTensorNet, cuStateVec,
                                    exts...) == []
end

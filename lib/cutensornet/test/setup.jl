using Test, CUDA, LinearAlgebra
using cuTENSOR
using cuTensorNet
import cuTensorNet: CuTensorNetwork, rehearse_contraction, perform_contraction!, gateSplit!, AutoTune, NoAutoTune

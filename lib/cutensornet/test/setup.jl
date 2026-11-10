using Test, CUDACore, LinearAlgebra
using cuTENSOR
using cuTensorNet
import cuTensorNet: rehearse_contraction, tensor_qr!, tensor_svd!, gate_split!

# compute descriptors are accessed through external symbols
for desc in [:CUTENSOR_COMPUTE_DESC_16F,
             :CUTENSOR_COMPUTE_DESC_16BF,
             :CUTENSOR_COMPUTE_DESC_TF32,
             :CUTENSOR_COMPUTE_DESC_3XTF32,
             :CUTENSOR_COMPUTE_DESC_32F,
             :CUTENSOR_COMPUTE_DESC_64F]
    @eval begin
        $desc() = cutensorComputeDescriptor_t(cglobal(($(QuoteNode(desc)), libcutensor)))
    end
end

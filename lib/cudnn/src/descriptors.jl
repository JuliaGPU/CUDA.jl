using Base: @__doc__


"""
    @cudnnDescriptor(XXX, setter=cudnnSetXXXDescriptor)

Defines a new type `cudnnXXXDescriptor` with a single field `ptr::cudnnXXXDescriptor_t` and
its constructor. The second optional argument is the function that sets the descriptor
fields and defaults to `cudnnSetXXXDescriptor`. The constructor is memoized, i.e. when
called with the same arguments it returns the same object rather than creating a new one.

The arguments of the constructor and thus the keys to the memoization cache depend on the
setter: If the setter has arguments `cudnnSetXXXDescriptor(ptr::cudnnXXXDescriptor_t,
args...)`, then the constructor has `cudnnXXXDescriptor(args...)`. The user can control
these arguments by defining a custom setter.
"""
macro cudnnDescriptor(x, set = Symbol("cudnnSet$(x)Descriptor"))
    sname = Symbol("cudnn$(x)Descriptor")
    tname = Symbol("cudnn$(x)Descriptor_t")
    cache = Symbol("cudnn$(x)DescriptorCache")
    cache_lock = Symbol("cudnn$(x)DescriptorCacheLock")
    create = Symbol("cudnnCreate$(x)Descriptor")
    destroy = Symbol("cudnnDestroy$(x)Descriptor")
    return quote
        @__doc__ mutable struct $sname                      # needs to be mutable for finalizer
            ptr::$tname
            $sname(p::$tname) = new(p)                      # prevent $sname(::Any) default constructor
            $sname(p::Ptr{Cvoid}) = new(p)                  # tests rely on passing C_NULL; why?
        end
        Base.unsafe_convert(::Type{<:Ptr}, d::$sname)=d.ptr # needed for ccalls
        const $cache = Dict{Tuple,$sname}()                 # Dict is 3x faster than IdDict!
        const $cache_lock = ReentrantLock()
        function $sname(args...)
            d = lock($cache_lock) do
                get($cache, args, nothing)
            end
            if d === nothing
                ptr = $tname[C_NULL]
                $create(ptr)
                $set(ptr[1], args...)
                d = $sname(ptr[1])
                finalizer(x->$destroy(x.ptr), d)
                lock($cache_lock) do
                    $cache[args] = d
                end
            end
            return d
        end
    end |> esc
end


"""
    cudnnDropoutDescriptor(dropout::Real)
"""
@cudnnDescriptor(Dropout, cudnnSetDropoutDescriptorFromFloat)


"""
    cudnnFilterDescriptor(dataType::cudnnDataType_t,
                          format::cudnnTensorFormat_t,
                          nbDims::Cint,
                          filterDimA::Vector{Cint})
"""
@cudnnDescriptor(Filter, cudnnSetFilterNdDescriptor)


"""
    cudnnRNNDescriptor(algo::cudnnRNNAlgo_t,
                       cellMode::cudnnRNNMode_t,
                       biasMode::cudnnRNNBiasMode_t,
                       dirMode::cudnnDirectionMode_t,
                       inputMode::cudnnRNNInputMode_t,
                       dataType::cudnnDataType_t,
                       mathPrec::cudnnDataType_t,
                       mathType::cudnnMathType_t,
                       inputSize::Int32,
                       hiddenSize::Int32,
                       projSize::Int32,
                       numLayers::Int32,
                       dropoutDesc::cudnnDropoutDescriptor_t,
                       auxFlags::UInt32)
"""
@cudnnDescriptor(RNN, cudnnSetRNNDescriptor_v8)


"""
    cudnnRNNDataDescriptor(dataType::cudnnDataType_t,
                           layout::cudnnRNNDataLayout_t,
                           maxSeqLength::Cint,
                           batchSize::Cint,
                           vectorSize::Cint,
                           seqLengthArray::Vector{Cint},
                           paddingFill::Ptr{Cvoid})
"""
@cudnnDescriptor(RNNData)


"""
    cudnnTensorDescriptor(format::cudnnTensorFormat_t,
                          dataType::cudnnDataType_t,
                          nbDims::Cint,
                          dimA::Vector{Cint})
"""
@cudnnDescriptor(Tensor, cudnnSetTensorNdDescriptorEx)

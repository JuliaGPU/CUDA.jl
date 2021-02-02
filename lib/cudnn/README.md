## High level interface to cuDNN functions
Deniz Yuret, Nov 6, 2020

The goal of the high-level interface is to map the low level cuDNN calls to more natural
Julia functions. Here are some design choices I followed:

**Naming:** We try to keep the same function, argument, and type names from the cuDNN
library in the high level interface. The wrappers for descriptors drop the `_t` suffix,
e.g. `cudnnPoolingDescriptor_t => cudnnPoolingDescriptor`.

**Descriptors:** The cuDNN functions take data and operator descriptors. Most of these
descriptors are relatively fast to create (~500 ns for a cudnnTensorDescriptor) so they may
not be worth preallocating for the user but we provide keyword options anyway. We cache
descriptors (~100 ns) so we can use them as hash keys for memoization, which also saves a
bit of memory and speed.  All descriptor fields are `isbits` types with the exception of the
`cudnnDropoutDescriptor` which points to a random number generator state and is used as a
field of some other descriptors.

**Operator descriptors:** Descriptors such as `cudnnPoolingDescriptor` specify the options
for an operator such as stride and padding. For operators with descriptors we have one
method that takes keyword arguments with reasonable defaults to construct the descriptor and
another method that takes a pre-initialized descriptor as its last argument.  This way a
casual user can call the first method without worrying about the descriptor format, only
specifying non-default options, whereas a layer architect can keep a preset descriptor in
the layer that gets passed to the function using the second method. We try to use generic
Julia types for keyword arguments that specify default descriptor fields and convert these
to the appropriate cudnn types during descriptor construction.

**Output arrays:** The low level cuDNN functions take pre-allocated output arrays. The high
level interface has one Julia function that allocates its own output array
(e.g. `cudnnPoolingForward`) and another with an exclamation mark that takes a pre-allocated
output array as its first argument (e.g. `cudnnPoolingForward!`).

**Methods:** Each cuDNN forward function may have up to four methods depending on whether
the descriptor and the output array are specified:

    cudnnPoolingForward(x; kwargs...)
    cudnnPoolingForward(x, d::cudnnPoolingDescriptor; kwargs...)
    cudnnPoolingForward!(y, x; kwargs...)
    cudnnPoolingForward!(y, x, d::cudnnPoolingDescriptor; kwargs...)

The conventional order of arguments for these public methods is:

    ([output], weights, inputs, [descriptor]; kwargs...)

**AD method:** Neither the high level nor the low level interface is sometimes
appropriate for gradient definitions, e.g. the low level API may not return a value, the
high level API may have some gradient target parameters as keyword arguments. To solve this
issue the API exposes an intermediate function with an AD suffix,
e.g. `cudnnPoolingForwardAD`, that is called by the high level method and that makes
the low level library call. These methods may not seem like they are doing anything useful,
but they should not be removed so automatic gradient packages may make use of them.

**Backward functions:** The point of a high level interface is to give the user appropriate
defaults for the many options of typical cudnn functions. Backward functions do not have
meaningful defaults because they need to copy their options from the corresponding forward
function. Therefore we do not need high level APIs for backward functions unless they are
useful in some other way. See Knet/src/cudnn for example uses.

**Types:** Do not specify types for array arguments. Leave the high level functions generic
so they can be called with CuArray, KnetArray, AutoGrad.Param etc. Types can and should be
specified for non-array arguments. In the API we use `nothing` to indicate unspecified array
argument values, convert these to `C_NULL` or `CU_NULL` as appropriate only at the low-level
call. Similarly for numbers the API should accept generic types like `Integer` or `Real` and
convert these to the appropriate specific type, e.g. `Cint` or `Cdouble` only at the
low-level call.

**Workspace:** Some functions need a temporary allocated workspace whose required size is
determined by another cudnn call. Unfortunately, the required size may depend on factors
other than the current inputs (see [this
issue](https://github.com/FluxML/Flux.jl/issues/923#issuecomment-558671966)), so the usage
of the `@workspace` macro is used at a point as close to the library call as possible. One
exception to this is cases where the same workspace will be passed to the backward call, in
which case we allocate a regular CuArray.

**Training vs Inference:** There is no consistent way cuDNN distinguishes training vs inference calls:
* BatchNormalization and Normalization have two separate functions: `cudnnNormalizationForwardTraining / Inference`
* RNN has an indicator argument: `fwdMode` in `cudnnRNNForward`
* MultiHeadAttn looks at the `reserveSpace` argument to decide: if `NULL` inference mode, otherwise training mode
* Dropout always runs in training mode with a non-NULL `reserveSpace` (it doesn't make sense in inference mode)
* Activation, convolution, pooling, softmax, optensor, addtensor, reducetensor do not make a distinction between the two modes

In the high level API we assume inference by default and let the gradient packages override when necessary.
See the gradient implementations in Knet/src/cudnn for examples.

**TODO:** 
* Keyword arg descriptor constructors.
* Test forw fns with descriptors: check for desc vs kwarg incompatibility.
* Find out about cudnnRNNSetClip_v8.
* Test with Knet.Ops20.
* Command used to test: julia17 --project -e 'using Pkg; Pkg.API.test(; test_args=`--memcheck --jobs=1 cudnn`)'

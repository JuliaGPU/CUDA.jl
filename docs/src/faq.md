# Frequently Asked Questions

This page is a compilation of frequently asked questions and answers.


## Can you wrap this or that CUDA API?

If a certain API isn't wrapped with some high-level functionality, you can always use the
underlying C APIs which are always available as unexported methods. For example, you can
access the CUDA driver library as `cu` prefixed, unexported functions like
`CUDA.cuDriverGetVersion`. Similarly, vendor libraries like CUBLAS are available through
their exported submodule handles, e.g., `CUBLAS.cublasGetVersion_v2`.

Any help on designing or implementing high-level wrappers for this low-level functionality
is greatly appreciated, so please consider contributing your uses of these APIs on the
respective repositories.

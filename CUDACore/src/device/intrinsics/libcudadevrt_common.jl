
# Automatically generated using Clang.jl

# manually cherry-picked from the generated wrappers, because we don't need the full extend
# of the API and can reuse some structures from the CUDA driver API
# TODO: aren't enums also reusable?

# compatible structures from the CUDA driver API
const dim3 = CuDim3
const cudaStream_t = CUstream
const cudaEvent_t = CUevent

@cenum cudaError::UInt32 begin
    cudaSuccess = 0
    cudaErrorInvalidValue = 1
    cudaErrorMemoryAllocation = 2
    cudaErrorInitializationError = 3
    cudaErrorCudartUnloading = 4
    cudaErrorProfilerDisabled = 5
    cudaErrorProfilerNotInitialized = 6
    cudaErrorProfilerAlreadyStarted = 7
    cudaErrorProfilerAlreadyStopped = 8
    cudaErrorInvalidConfiguration = 9
    cudaErrorInvalidPitchValue = 12
    cudaErrorInvalidSymbol = 13
    cudaErrorInvalidHostPointer = 16
    cudaErrorInvalidDevicePointer = 17
    cudaErrorInvalidTexture = 18
    cudaErrorInvalidTextureBinding = 19
    cudaErrorInvalidChannelDescriptor = 20
    cudaErrorInvalidMemcpyDirection = 21
    cudaErrorAddressOfConstant = 22
    cudaErrorTextureFetchFailed = 23
    cudaErrorTextureNotBound = 24
    cudaErrorSynchronizationError = 25
    cudaErrorInvalidFilterSetting = 26
    cudaErrorInvalidNormSetting = 27
    cudaErrorMixedDeviceExecution = 28
    cudaErrorNotYetImplemented = 31
    cudaErrorMemoryValueTooLarge = 32
    cudaErrorInsufficientDriver = 35
    cudaErrorInvalidSurface = 37
    cudaErrorDuplicateVariableName = 43
    cudaErrorDuplicateTextureName = 44
    cudaErrorDuplicateSurfaceName = 45
    cudaErrorDevicesUnavailable = 46
    cudaErrorIncompatibleDriverContext = 49
    cudaErrorMissingConfiguration = 52
    cudaErrorPriorLaunchFailure = 53
    cudaErrorLaunchMaxDepthExceeded = 65
    cudaErrorLaunchFileScopedTex = 66
    cudaErrorLaunchFileScopedSurf = 67
    cudaErrorSyncDepthExceeded = 68
    cudaErrorLaunchPendingCountExceeded = 69
    cudaErrorInvalidDeviceFunction = 98
    cudaErrorNoDevice = 100
    cudaErrorInvalidDevice = 101
    cudaErrorStartupFailure = 127
    cudaErrorInvalidKernelImage = 200
    cudaErrorDeviceUninitilialized = 201
    cudaErrorMapBufferObjectFailed = 205
    cudaErrorUnmapBufferObjectFailed = 206
    cudaErrorArrayIsMapped = 207
    cudaErrorAlreadyMapped = 208
    cudaErrorNoKernelImageForDevice = 209
    cudaErrorAlreadyAcquired = 210
    cudaErrorNotMapped = 211
    cudaErrorNotMappedAsArray = 212
    cudaErrorNotMappedAsPointer = 213
    cudaErrorECCUncorrectable = 214
    cudaErrorUnsupportedLimit = 215
    cudaErrorDeviceAlreadyInUse = 216
    cudaErrorPeerAccessUnsupported = 217
    cudaErrorInvalidPtx = 218
    cudaErrorInvalidGraphicsContext = 219
    cudaErrorNvlinkUncorrectable = 220
    cudaErrorJitCompilerNotFound = 221
    cudaErrorInvalidSource = 300
    cudaErrorFileNotFound = 301
    cudaErrorSharedObjectSymbolNotFound = 302
    cudaErrorSharedObjectInitFailed = 303
    cudaErrorOperatingSystem = 304
    cudaErrorInvalidResourceHandle = 400
    cudaErrorIllegalState = 401
    cudaErrorSymbolNotFound = 500
    cudaErrorNotReady = 600
    cudaErrorIllegalAddress = 700
    cudaErrorLaunchOutOfResources = 701
    cudaErrorLaunchTimeout = 702
    cudaErrorLaunchIncompatibleTexturing = 703
    cudaErrorPeerAccessAlreadyEnabled = 704
    cudaErrorPeerAccessNotEnabled = 705
    cudaErrorSetOnActiveProcess = 708
    cudaErrorContextIsDestroyed = 709
    cudaErrorAssert = 710
    cudaErrorTooManyPeers = 711
    cudaErrorHostMemoryAlreadyRegistered = 712
    cudaErrorHostMemoryNotRegistered = 713
    cudaErrorHardwareStackError = 714
    cudaErrorIllegalInstruction = 715
    cudaErrorMisalignedAddress = 716
    cudaErrorInvalidAddressSpace = 717
    cudaErrorInvalidPc = 718
    cudaErrorLaunchFailure = 719
    cudaErrorCooperativeLaunchTooLarge = 720
    cudaErrorNotPermitted = 800
    cudaErrorNotSupported = 801
    cudaErrorSystemNotReady = 802
    cudaErrorSystemDriverMismatch = 803
    cudaErrorCompatNotSupportedOnDevice = 804
    cudaErrorStreamCaptureUnsupported = 900
    cudaErrorStreamCaptureInvalidated = 901
    cudaErrorStreamCaptureMerge = 902
    cudaErrorStreamCaptureUnmatched = 903
    cudaErrorStreamCaptureUnjoined = 904
    cudaErrorStreamCaptureIsolation = 905
    cudaErrorStreamCaptureImplicit = 906
    cudaErrorCapturedEvent = 907
    cudaErrorStreamCaptureWrongThread = 908
    cudaErrorUnknown = 999
    cudaErrorApiFailureBase = 10000
end
const cudaError_t = cudaError

const cudaStreamDefault = 0x00
const cudaStreamNonBlocking = 0x01

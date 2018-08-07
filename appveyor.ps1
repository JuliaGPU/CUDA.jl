Set-PSDebug -Trace 1
$ErrorActionPreference = "Stop"


# If there's a newer build queued for the same PR, cancel this one

if ($env:APPVEYOR_PULL_REQUEST_NUMBER -and $env:APPVEYOR_BUILD_NUMBER -ne ((Invoke-RestMethod `
        https://ci.appveyor.com/api/projects/$env:APPVEYOR_ACCOUNT_NAME/$env:APPVEYOR_PROJECT_SLUG/history?recordsNumber=50).builds | `
        Where-Object pullRequestId -eq $env:APPVEYOR_PULL_REQUEST_NUMBER)[0].buildNumber) { `
    throw "There are newer queued builds for this pull request, failing early."
}


# Julia

$julia_installers = @{}
$julia_installers.Add('0.7',     'https://julialang-s3.julialang.org/bin/winnt/x64/0.7/julia-0.7.0-rc2-win64.exe')
$julia_installers.Add('nightly', 'https://julialangnightlies-s3.julialang.org/bin/winnt/x64/julia-latest-win64.exe')

$julia_installer = $julia_installers.Get_Item($env:JULIA)
Start-FileDownload $julia_installer -FileName julia.exe

.\julia.exe /S /D=C:\julia


# CUDA

$cuda_installers = @{}
$cuda_installers.Add('8.0', 'https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_windows-exe')
$cuda_installers.Add('9.0', 'https://developer.nvidia.com/compute/cuda/9.0/prod/local_installers/cuda_9.0.176_windows-exe')
$cuda_installers.Add('9.1', 'https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows')
$cuda_installers.Add('9.2', 'https://developer.nvidia.com/compute/cuda/9.2/Prod/local_installers/cuda_9.2.88_windows')

$cuda_installer = $cuda_installers.Get_Item($env:CUDA)
Start-FileDownload $cuda_installer -FileName cuda.exe

$cuda_components = @{}
$cuda_components.Add('8.0', @("compiler_8.0", "cudart_8.0"))
$cuda_components.Add('9.0', @("compiler_9.0", "cudart_9.0"))
$cuda_components.Add('9.1', @("nvcc_9.1", "cudart_9.1"))
$cuda_components.Add('9.2', @("nvcc_9.2", "cudart_9.2"))

$components = $cuda_components.Get_Item($env:CUDA)
.\cuda.exe -s @components | Out-Null

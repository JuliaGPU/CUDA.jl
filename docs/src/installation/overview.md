# [Overview](@id InstallationOverview)

The Julia CUDA stack requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and matching [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). For now, both of these components
should be manually installed. If you are a Linux user, you should consider installing these
dependencies using a package manager instead of downloading them from the NVIDIA homepage;
refer to your distribution's documentation for more details.

To make sure you have everything set-up, you can try executing some of the applications that
the driver and toolkit provide. On Linux, you can verify driver availability by executing
`nvidia-smi`, and you have installed CUDA successfully if you can execute `ptxas --version`.

## Installation of CUDA toolkit
If you had problem installing CUDA, we will guide you through the process:

- System Requirements
  -------------------

  - Check if your system has [CUDA-capable GPU](https://developer.nvidia.com/cuda-gpus) or by entering `lspci | grep -i nvidia`. If you don't see anything, update PCI hardware database by `update-pciids` and re-run the previous `lspci` command.

  - Check if you have supported version of linux by entering `uname -m` and `cat /etc/*release`.

  - Check if the system has correct Kernel Headers and Development Packages installed by entering `uname -r`. If not installed, then you can install them by `sudo apt-get install linux-headers-$(uname -r)` on Ubuntu. For other linux distributions check the installation page.

  - Check if your [GCC version](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) `gcc --version` is compatible with the version of CUDA toolkit you are about to install.

    - If you don't have compatible GCC version, then you can either build from source (not recommended) or follow the below mentioned steps:

      - `sudo apt-get install gcc-xx g++-xx cc-xx`
      - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-x 100 --slave /usr/bin/g++ g++ /usr/bin/g++-x`
      - `sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-xx 50 --slave /usr/bin/g++ g++ /usr/bin/g++-xx`

  - If any previous NVIDIA CUDA toolkit or drivers are present purge them by:

  `sudo apt-get remove --purge '^nvidia-.*'`
  `sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl`
  `sudo /usr/bin/nvidia-uninstall`
  `sudo apt-get --purge remove <package_name>`

  - If Nouveau drivers are present, then disable them otherwise it will create problems while downloading NVIDIA drivers by:

  `sudo stop lightdm`
  Create a file:
  `sudo nano /etc/modprobe.d/blacklist-nouveau.conf`
  Write both the lines in the file:
  `blacklist nouveau`
  `options nouveau modeset=0`
  Then update the initramfs and don't forget to reboot:
  `sudo update-initramfs -u`
  `sudo reboot`

  - Check for the NVIDIA drivers availability by executing `nvidia-smi`. If driver is not installed then you can find for your [compatible driver version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver) and [download](https://www.nvidia.com/Download/index.aspx) and install it. If any propriety drivers are in use, it is strongly recommend not to use any of those.

- Download CUDA Toolkit
  ---------------------

  - Download the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). It is recommended to use distribution-specific packages (RPM and Deb packages) rather than distribution-independent packages(runfile packages). Distribution-independent packages has the advantage of working across a wider set of linux distributions but doesn't update the distribution's native package management system. Follow the mentioned instructions to install CUDA and then follow the EULA instructions.

- Post-Installation
  -----------------

- Add the path of installed CUDA package to the PATH variable:

`export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}`

- Change the environment variables for 32-bit/64-bit operating systems:

`export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

or

`export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`

- Install Writable Samples:

`cuda-install-samples-10.2.sh <dir>`

- Verify the installation:

  - Verify the driver version
  
  `cat /proc/driver/nvidia/version`

  - Compiling the examples

  `nvcc -V` to check the version of CUDA Toolkit.
  `cd ~/NVIDIA_CUDA-10.2_Samples`
  `make`

  - Running the binaries

  `./deviceQuery` If Result = PASS appears then CUDA is installed properly.

## CUDA discovery

Once you've installed the NVIDIA driver and CUDA toolkit, the Julia CUDA packages should
automatically pick up your installation by means of the functionality in CUDAapi.jl. Some
guidelines to make sure this works:

- CUDA driver: the driver library should be loadable with Libdl (e.g.,
  `Libdl.dlopen("libcuda")`)
- CUDA toolkit: the CUDA binaries should be on `PATH`

Alternatively, you can use the `CUDA_HOME` environment variable to point to an installation
of the CUDA toolkit.

To debug this, set `JULIA_DEBUG=CUDAapi` (or more generally `JULIA_DEBUG=all`) for details
on which paths are probed. If you file an issue, always include this information.


### Multiple CUDA toolkits

Generally, multiple installed CUDA toolkits are no supported because this may lead to
incompatible libraries being picked up. However, if you use the `CUDA_HOME` environment
variable to point to an installation, all other discovery heuristics will be disabled. This
should result in only that version of the CUDA toolkit being used, on the condition no other
toolkit is present in the global environment (`PATH`, `LD_LIBRARY_PATH`).


## Version compatibility

You should always use a CUDA toolkit that is supported by your driver. That means that the
toolkit version number should be lower or equal than the CUDA API version that is supported
by your driver (only taking into account the major and minor part of the version number).

Both these versions can be queried using the tools mentioned above, but you can also use the
Julia packages:

```julia
julia> using CUDAdrv, CUDAnative

julia> CUDAdrv.version() # CUDA toolkit supported by the driver
v"10.2.0"

julia> CUDAnative.version() # CUDA toolkit installed
v"10.2.89"
```

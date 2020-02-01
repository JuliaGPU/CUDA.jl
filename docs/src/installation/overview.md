# [Overview](@id InstallationOverview)

The Julia CUDA stack requires users to have a functional [NVIDIA
driver](https://www.nvidia.com/Download/index.aspx) and matching [CUDA
toolkit](https://developer.nvidia.com/cuda-downloads). For now, both of these components
should be manually installed. If you are a Linux user, you should consider installing these
dependencies using a package manager instead of downloading them from the NVIDIA homepage;
refer to your distribution's documentation for more details.

[CUDA installation docs](https://docs.nvidia.com/cuda/index.html#installation-guides) are the best source and step-by-step guide for CUDA toolkit and driver installation.

CUDA toolkit consists of two main components:

- the CUDA runtime development libraries
- the driver components

The driver components are further divided into:

- the kernel mode components (the 'display' driver)
- the user mode components (the CUDA driver, the OpenGL driver, etc.)

The interlink between CUDA runtime libraries and CUDA driver (especially `libcuda.so`) forces the user to update the entire driver stack to use the latest CUDA software. Starting with CUDA 10.0, NVIDIA introduced a new forward-compatible upgrade path that allows the kernel mode components on the system to remain untouched, while the CUDA driver is upgraded and the path minimizes the risks associated with new driver deployments. Both the CUDA driver and the CUDA runtime are not source compatible across the different SDK releases. APIs can be deprecated and removed, requiring changes to the application. Although driver APIs can change, they are versioned so they will function on a minimum supported driver.

The CUDA Toolkit can be installed using either of two different installation mechanisms:

- distribution-specific packages (RPM and Deb packages)

The Package Manager installation interfaces with your system's package management system. If those packages are available in an online repository, they will be automatically downloaded in a later step. Otherwise, the repository package also installs a local repository containing the installation packages on the system. 

- distribution-independent package (runfile packages)

The Runfile installation installs the NVIDIA Driver, the CUDA Toolkit, and CUDA Samples, via an interactive ncurses-based interface. The Runfile installation does not include support for cross-platform development. 

The distribution-independent package has the advantage of working across a wider set of Linux distributions, but does not update the distribution's native package management system. The distribution-specific packages interface with the distribution's native package management system. It is recommended to use the distribution-specific packages, where possible.

`lsmod` formats the contents of the `/proc/modules`, showing what kernel modules are currently loaded. `lsmod | grep nvidia` shows which all NVIDIA modules are loaded. If not then you need to install the NVIDIA driver.

Before that check for your GPU by executing `nvidia-smi`. Find the appropriate driver for your GPU. `dmesg` obtains its data by reading the kernel ring buffer and is useful when troubleshooting or obtaining information about the hardware on a system. If the `dmesg` doesn't contain any information about NVIDIA then you need to add another driver.

You need to install the NVIDIA driver before installing the CUDA toolkit. Before installing the driver, exit the X server and terminate all OpenGL applications. You should also set the default run level on your system such that it will boot to a VGA console, and not directly to X. Doing so will make it easier to recover if there is a problem during the installation process. If NVIDIA installer is being installed on a system that is set up to use the Nouveau driver, then you should first disable it. Build dependencies of nvidia-installer including `ncurses pciutils`. `nvidia-installer` will install itself to `/usr/bin/nvidia-installer`, which may be used to uninstall drivers, auto-download updated drivers. NVIDIA drivers can be downloaded from the [NVIDIA website](http://www.nvidia.com). Check [NVIDIA driver installation documentation](https://download.nvidia.com/XFree86/Linux-x86_64/410.104/README/introduction.html) for further details.

User interface shared libraries are built into `nvidia-installer` to avoid potential problems with
the installer not being able to find the user interface libraries (or finding the wrong ones, etc) after the shared library is built, the utility `gen-ui-array` is run on it to create a source file storing a byte array of the data that is the shared library.  When the installer runs, it writes this byte data to a temporary file and then `dlopens()` it. It contains source for a simple tool `mkprecompiled`, which is a standalone utility that can be used to build a precompiled kernel interface package independently of `nvidia-installer` and is used for stamping nv-linux.o's (aka the kernel interfaces) with the necessary information so that the installer can match it to a running kernel. `nvidia-installer` can automate the process of building a precompiled kernel interface package when used with the installer's `--add-this-kernel` option.

To make sure you have everything set-up, you can try executing some of the applications that
the driver and toolkit provide. On Linux, you can verify driver availability by executing
`nvidia-smi`, and you have installed CUDA successfully if you can execute `ptxas --version` or `nvcc -V` and if doesn't display anything, you need to install CUDA toolkit.

If you are still not sure if CUDA is installed and PATH is set, try running a C hello world program in CUDA which is as follows:

```
#include<stdio.h>
__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>(); 
    return 0;
}

```
Compile it by executing `nvcc hello.cu -o hello` in a bourne shell. If the program compiles without any errors, then CUDA is installed properly in your system.

JULIA uses the CUDA driver to compile PTX code to GPU assembly, whereas nvcc does this upfront. So even though your CUDA set-up might work fine with CUDA C, it doesn't with CUDAnative because of using JIT compiling code with the driver. CUDA also needs to be compatible with `sm_20` for the specific GPU-architecture on the system.

In case there is not enough GPU cards available, you can submit passive jobs, using `sbatch`. You can compile CUDA applications on a node without GPU, using the same modules. Reserve a node with one GPU for interactive development, load the necessary modules, and save them for a quick restore.
```

> srun -p gpu  -n1 -c1 --gres=gpu:1 --pty bash -i
$ module av cuda compiler/gcc
$ module load compiler/LLVM system/CUDA
$ nvidia-smi
$ module save cuda

```

Below is an example of `sbatch` file : 

```

#!/bin/bash -l
#SBATCH --job-name="GPU build"
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=qos-gpu

if [ -z "$1" ]
then
    echo "Missing required source (.cu), and optional execution arguments."
    exit
fi

src=${1}
exe=$(basename ${1/cu/out})
ptx=$(basename ${1/cu/ptx})
prf=$(basename ${1/cu/prof})
shift
args=$*

# after the module profile is saved (see above)
module restore cuda

# compile
srun nvcc -arch=compute_70 -o ./$exe $src
# save ptx
srun nvcc -ptx -arch=compute_70 -o ./$ptx $src
# execute
srun ./$exe $args
# profile
srun nvprof --log-file ./$prf ./$exe $args
echo "file: $prf"
cat ./$prf

```

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

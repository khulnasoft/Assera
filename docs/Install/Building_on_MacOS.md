[//]: # (Project: Assera)
[//]: # (Version: v1.2)

## Installing on MacOS

### Install Dependencies

Assera requires the following tools and libraries:

* A C++ compiler that supports C++ 17, such as `clang`, which is bundled in XCode
* CMake 3.14 or newer
* Python 3.7 or newer
* Ninja
* Ccache
* LLVM OpenMP 5, if using parallelization

Homebrew is a package manager that makes it easy to install the prerequisites. Homebrew can be downloaded and installed by:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

If you already have Homebrew installed, update it to the latest version by typing:

```
brew update
```

Install the dependencies:

Intel MacOS|Apple Silicon|
|--|--|
|`brew install cmake python ninja-build ccache libomp pkg-config`|`brew install cmake python ninja ccache libomp pkg-config`

#### Clang

Select the `clang` compiler from XCode:

```
xcode-select --install
```

### Clone Assera

A version of [git](https://git-scm.com/download) should already be included in XCode.

Clone the git repository:

```
git clone --recurse-submodules https://github.com/khulnasoft/Assera
```

### Build and install Assera

Run the `build.sh` script to install dependencies and build the Assera Python package (replace `<path_to_assera>` with the path to the cloned Assera repository).

```shell
cd <path_to_assera>
sh ./build.sh
```

Update or install the resulting `.whl` file from the `dist` sudirectory. The name depends on your Python version, your OS and your CPU architecture.
```shell
pip install -U ./dist/assera-0.0.1-cp37-cp37-macosx_10_15_x86_64.whl --find-links=dist
```

### Build and install using CMake

Assera can also be built using CMake (intended for expert users).

#### Install dependencies

```shell
cd <path_to_assera>
git submodule init
git submodule update
./external/vcpkg/bootstrap-vcpkg.sh
./external/vcpkg/vcpkg install catch2 tomlplusplus assera-llvm --overlay-ports=external/llvm
```

The last command typically takes a few hours to build and then install Assera's fork of LLVM. We recommend reserving at least 20GB of disk space for the LLVM build.

#### Configure CMake

```shell
cd <path_to_assera>
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja
```

#### Build and run tests

```shell
cmake --build . --config Release
ctest -C Release
```

#### Install

```shell
cmake --build . --config Release --target install
```
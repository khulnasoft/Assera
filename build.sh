#!/bin/sh

set -e

# Build script for the Assera Python package
ASSERA_ROOT=`pwd`

# Ensure that submodules are cloned
git submodule init
git submodule update

# Install dependencies
# Linux: apt get install pkg-config
pip install -r requirements.txt
cd external/vcpkg
./bootstrap-vcpkg.sh
./vcpkg install catch2 tomlplusplus --overlay-ports=../llvm

if [ -z "${LLVM_SETUP_VARIANT}" ] && [ -f "$ASSERA_ROOT/CMake/LLVMSetupConan.cmake" ]; then
    echo Using LLVM from Conan
    export LLVM_SETUP_VARIANT=Conan
else 
    echo Using LLVM from vcpkg
    export LLVM_SETUP_VARIANT=Default

    # Uncomment these lines below to build a debug version (will include release as well, due to vcpkg quirks)
    # export LLVM_BUILD_TYPE=debug
    # export VCPKG_KEEP_ENV_VARS=LLVM_BUILD_TYPE

    # Install LLVM (takes a couple of hours and ~20GB of space)
    ./vcpkg install assera-llvm --overlay-ports=../llvm
fi

# Build the assera package
cd "$ASSERA_ROOT"
python setup.py build bdist_wheel

# Build the subpackages
cd "$ASSERA_ROOT/assera/python/compilers"
python setup.py build bdist_wheel -d "$ASSERA_ROOT/dist"
cd "$ASSERA_ROOT/assera/python/llvm"
python setup.py build bdist_wheel -d "$ASSERA_ROOT/dist"
cd "$ASSERA_ROOT/assera/python/gpu"
python setup.py build bdist_wheel -d "$ASSERA_ROOT/dist"
cd "$ASSERA_ROOT"

echo Complete. Packages are in the 'dist' folder.

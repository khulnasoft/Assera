@echo off
REM ####################################################################################################

setlocal

set ASSERA_ROOT=%~dp0

REM Ensure that submodules are cloned
git submodule update --init --recursive
git pull --recurse-submodules

REM Install dependencies
pip install -r requirements.txt
cd external\vcpkg
call bootstrap-vcpkg.bat
vcpkg install catch2:x64-windows tomlplusplus:x64-windows --overlay-ports=..\llvm

if exist "%ASSERA_ROOT%\CMake\LLVMSetupConan.cmake" (
    echo Using LLVM from Conan
    set LLVM_SETUP_VARIANT=Conan
) else (
    echo Using LLVM from vcpkg
    set LLVM_SETUP_VARIANT=Default

    REM Uncomment these lines below to build a debug version (will include release as well, due to vcpkg quirks)
    REM set LLVM_BUILD_TYPE=debug
    REM set VCPKG_KEEP_ENV_VARS=LLVM_BUILD_TYPE

    REM Install LLVM (takes a couple of hours and ~20GB of space)
    vcpkg install assera-llvm:x64-windows --overlay-ports=..\llvm
)

REM Build the assera package
cd "%ASSERA_ROOT%"
python setup.py build bdist_wheel

REM Build the subpackages
cd "%ASSERA_ROOT%\assera\python\compilers"
python setup.py build bdist_wheel -d "%ASSERA_ROOT%\dist"
cd "%ASSERA_ROOT%\assera\python\llvm"
python setup.py build bdist_wheel -d "%ASSERA_ROOT%\dist"
cd "%ASSERA_ROOT%\assera\python\gpu"
python setup.py build bdist_wheel -d "%ASSERA_ROOT%\dist"
cd "%ASSERA_ROOT%"

echo Complete. Packages are in the '%ASSERA_ROOT%\dist' folder.

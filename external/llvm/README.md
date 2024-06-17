# Assera LLVM vcpkg Port

This vcpkg LLVM port is based on the [official LLVM port](https://github.com/microsoft/vcpkg/blob/master/ports/llvm), but only builds the projects and targets needed for Assera.

## Build instructions

1. Ensure that you have at least 20GB of disk space

2. Bootstrap vcpkg:

    Windows

    ```shell
    cd <path_to_assera>\external\vcpkg
    call bootstrap-vcpkg.bat
    ```

    Linux / macOS:

    ```shell
    cd <path_to_assera>/external/vcpkg
    ./bootstrap-vcpkg.sh
    ```

3. Build and install LLVM:

    Windows

    ```shell
    cd <path_to_assera>
    externa\vcpkg\vcpkg.exe install assera-llvm:x64-windows --overlay-ports=external\llvm
    ```

    Linux / macOS:

    ```shell
    cd <path_to_assera>
    external/vcpkg/vcpkg install assera-llvm --overlay-ports=external/llvm
    ```

## Troubleshooting

To force a rebuild of a package: `external/vcpkg/vcpkg remove assera-llvm`


To reset vcpkg if there are issues with caching stale versions of packages:

1. Delete the vcpkg folder
2. `git submodule deinit -f .`
3. `git submodule update --init`
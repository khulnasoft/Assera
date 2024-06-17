####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_directories(${ASSERA_LIBRARIES_DIR})

add_subdirectory(nest)
add_subdirectory(exec)
add_subdirectory(assera)
add_subdirectory(value)
add_subdirectory(intrinsics)

add_subdirectory(argo)

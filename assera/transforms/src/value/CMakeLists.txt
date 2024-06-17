####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_directories(${ASSERA_LIBRARIES_DIR})
set(LLVM_TARGET_DEFINITIONS ValueConversion.td)
mlir_tablegen(ValueConversion.inc -gen-rewriters)
add_public_tablegen_target(ValueConversionIncGen)

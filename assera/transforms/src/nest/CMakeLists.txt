####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

include_directories(${ASSERA_LIBRARIES_DIR})
set(LLVM_TARGET_DEFINITIONS LoopNestToValue.td)
mlir_tablegen(LoopNestToValue.inc -gen-rewriters)
add_public_tablegen_target(LoopNestToValueIncGen)

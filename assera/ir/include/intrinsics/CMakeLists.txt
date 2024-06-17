####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

add_mlir_dialect(AsseraIntrinsics accintr)

set(LLVM_TARGET_DEFINITIONS AsseraIntrinsics.td)
mlir_tablegen(AsseraIntrinsicsConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(AsseraIntrinsicsConversionsIncGen)

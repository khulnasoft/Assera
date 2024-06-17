####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS AsseraPasses.td)
mlir_tablegen(AsseraPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(AsseraConversionIncGen)

####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(LLVM_TARGET_DEFINITIONS AsseraOps.td)
mlir_tablegen(AsseraOps.h.inc -gen-op-decls)
mlir_tablegen(AsseraOps.cpp.inc -gen-op-defs)
mlir_tablegen(AsseraDialect.h.inc -gen-dialect-decls -dialect=assera)
mlir_tablegen(AsseraDialect.cpp.inc -gen-dialect-defs -dialect=assera)
add_public_tablegen_target(AsseraOpsIncGen)

add_assera_ir_doc(AsseraOps -gen-dialect-doc AsseraDialect references/markdown/)

add_mlir_translation_library(IntrinsicToLLVMIRTranslation
  IntrinsicToLLVMIRTranslation.cpp

  ADDITIONAL_HEADER_DIRS
  ${ASSERA_BIN_DIR}/assera/ir/include

  DEPENDS
  MLIRAsseraIntrinsics
  AsseraIntrinsicsConversionsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRAsseraIntrinsics
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  )

target_include_directories(IntrinsicToLLVMIRTranslation PUBLIC
    ${ASSERA_BIN_DIR}/ir/include
)

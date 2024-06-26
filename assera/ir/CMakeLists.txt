####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# setup for using LLVM and MLIR
list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

get_filename_component(IR_INCLUDE include ABSOLUTE)
add_subdirectory(include)

set(src
    src/AffineConstraintsHelper.cpp
    src/DialectRegistry.cpp
    src/InitializeAssera.cpp
    src/IRUtil.cpp
    src/Metadata.cpp
    src/TranslateToHeader.cpp
    )

set(include
    include/AffineConstraintsHelper.h
    include/Common.td
    include/DialectRegistry.h
    include/InitializeAssera.h
    include/IRUtil.h
    include/Metadata.h
    include/TranslateToHeader.h
    )

set(intrinsics_src
    src/intrinsics/AsseraIntrinsicsDialect.cpp
    )
set(intrinsics_include
    include/intrinsics/AsseraIntrinsicsDialect.h)

set(accvalue_src
    src/value/ValueDialect.cpp
    src/value/ValueCanonicalization.cpp
    src/value/ValueFuncOpUtilities.cpp
    )

set(accvalue_include
    include/value/ValueDialect.h
    include/value/ValueAttributes.h
    include/value/ValueEnums.h
    include/value/ValueFuncOp.h
    include/value/ValueMMAOp.h
    include/value/ValueRangeOp.h
    )

set(accnest_src
    src/nest/AffineConstraints.cpp
    src/nest/AffineExpression.cpp
    src/nest/Index.cpp
    src/nest/IndexRange.cpp
    src/nest/IterationDomain.cpp
    src/nest/LoopNestAffineConstraints.cpp
    src/nest/LoopNestAttributes.cpp
    src/nest/LoopNestBuilder.cpp
    src/nest/LoopNestOps.cpp
    src/nest/LoopNestTypes.cpp
    src/nest/LoopVisitSchedule.cpp
    src/nest/OperandIndex.cpp
    src/nest/Range.cpp
    src/nest/TransformedDomain.cpp
    )

set(accnest_include
    include/nest/AffineConstraints.h
    include/nest/AffineExpression.h
    include/nest/Index.h
    include/nest/IndexRange.h
    include/nest/IterationDomain.h
    include/nest/LoopIndexInfo.h
    include/nest/LoopNestAffineConstraints.h
    include/nest/LoopNestAttributes.h
    include/nest/LoopNestBuilder.h
    include/nest/LoopNestOps.h
    include/nest/LoopNestInterfaces.h
    include/nest/LoopNestTypes.h
    include/nest/LoopVisitSchedule.h
    include/nest/OperandIndex.h
    include/nest/Range.h
    include/nest/TransformedDomain.h
    )

set(accexec_src
    src/exec/CacheAccessMaps.cpp
    src/exec/ExecutionPlanAttributes.cpp
    src/exec/ExecutionPlanOps.cpp
    )

set(accexec_include
    include/exec/CacheAccessMaps.h
    include/exec/ExecutionOptions.h
    include/exec/ExecutionPlanAttributes.h
    include/exec/ExecutionPlanOps.h
    include/exec/VectorizationInfo.h
    )

set(assera_src src/assera/AsseraOps.cpp)
set(assera_include include/assera/AsseraOps.h)

set(argo_src
    src/argo/ArgoOps.cpp
    src/argo/ArgoTypes.cpp
    src/argo/Utils.cpp
    )

set(argo_include
    include/argo/ArgoOps.h
    include/argo/ArgoTraits.h
    include/argo/ArgoTypes.h
    include/argo/Utils.h
    )

add_mlir_dialect_library(MLIRAsseraIntrinsics # This is an assera dialect, but the add_mlir_dialect() cmake function prepends "MLIR"
    ${intrinsics_src}

    ADDITIONAL_HEADER_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    DEPENDS
    MLIRAsseraIntrinsicsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )

InstallAsseraLibrary(MLIRAsseraIntrinsics)

# This is supposed to be overriden on the command line As of LLVM 8.0.1, the
# possible values within the list are: AArch64 AMDGPU ARM BPF Hexagon Lanai Mips
# MSP430 NVPTX PowerPC Sparc SystemZ WebAssembly X86 XCore
set(LLVM_EMITTER_TARGETS
    "X86;ARM;AMDGPU"
    CACHE
    STRING
    "List of LLVM emitter targets to support. Default is \"X86;ARM;AMDGPU\". Specify 'ALL' to support all targets"
    )
if (LLVM_EMITTER_TARGETS STREQUAL "ALL")
  set(LLVM_EMITTER_TARGETS_FINAL ${LLVM_ALL_TARGETS})
else ()
  set(LLVM_EMITTER_TARGETS_FINAL ${LLVM_EMITTER_TARGETS})
endif ()

set(emitter_targets_content "")
set(llvm_emitter_target_libs)
foreach (LLVM_EMITTER_TARGET ${LLVM_EMITTER_TARGETS_FINAL})
  if (NOT ${LLVM_EMITTER_TARGET} IN_LIST LLVM_ALL_TARGETS)
    message(
        FATAL_ERROR
        "Unrecognized LLVM emitter target: ${LLVM_EMITTER_TARGET}.\n\nTargets must be one of: ${LLVM_ALL_TARGETS}"
    )
  endif ()
  set(emitter_targets_content
      "${emitter_targets_content}    EMITTER_TARGET_ACTION(${LLVM_EMITTER_TARGET}) \\\n"
      )
  set(llvm_emitter_target_libs
      ${llvm_emitter_target_libs}
      LLVM${LLVM_EMITTER_TARGET}CodeGen
      LLVM${LLVM_EMITTER_TARGET}AsmParser
      LLVM${LLVM_EMITTER_TARGET}Disassembler
      # LLVM${LLVM_EMITTER_TARGET}AsmPrinter
      LLVM${LLVM_EMITTER_TARGET}Desc
      LLVM${LLVM_EMITTER_TARGET}Info
      )
endforeach (LLVM_EMITTER_TARGET LLVM_EMITTER_TARGETS)
configure_file(include/LLVMEmitterTargets.h.in build/LLVMEmitterTargets.h @ONLY)

set(library_name ir)
set(src
    ${src}
    ${accvalue_src}
    ${accnest_src}
    ${accexec_src}
    ${assera_src}
    ${argo_src}
    ${intrinsics_src}
    )

set(include
    ${include}
    ${accvalue_include}
    ${accnest_include}
    ${accexec_include}
    ${assera_include}
    ${argo_include}
    ${intrinsics_include}
    build/LLVMEmitterTargets.h
    )

add_library(${library_name} ${src} ${include})
target_include_directories(
    ${library_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} include
    PUBLIC
    $<BUILD_INTERFACE:${ASSERA_LIBRARIES_DIR}>
    $<INSTALL_INTERFACE:include/assera>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include/assera/ir/include>
)

target_include_directories(
    MLIRAsseraIntrinsics PRIVATE ${CMAKE_CURRENT_BINARY_DIR} include
    PUBLIC
    $<BUILD_INTERFACE:${ASSERA_LIBRARIES_DIR}>
    $<INSTALL_INTERFACE:include/assera>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include/assera/ir/include>
)

target_include_directories(${library_name} SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
    ${library_name}
    PUBLIC utilities
    MLIRSupport
    MLIRIR
    MLIRLinalg
    MLIRParser
    tomlplusplus::tomlplusplus
)

add_dependencies(
    ${library_name}
    ExecutionPlanOpsIncGen
    ExecutionPlanAttrsIncGen
    ExecutionPlanEnumsIncGen
    LoopNestExportedInterfacesIncGen
    LoopNestInterfacesIncGen
    LoopNestAttrsIncGen
    LoopNestOpsIncGen
    LoopNestToValueIncGen
    AsseraConversionIncGen
    AsseraOpsIncGen
    ValueAttrsIncGen
    ValueOpsIncGen
    MLIRAsseraIntrinsicsIncGen
    MLIRAsseraIntrinsics

    ArgoOpsIncGen
    ArgoStructuredOpsIncGen
    ArgoStructuredOpsInterfaceIncGen
)

#
# Add tests
#
add_subdirectory(test)

#
# Install headers and library
#
InstallAsseraHeaders(
    INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_LIST_DIR}/include
)
InstallAsseraLibrary(${library_name})

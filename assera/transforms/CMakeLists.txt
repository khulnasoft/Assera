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

get_filename_component(TRANSFORMS_INCLUDE include ABSOLUTE)
get_filename_component(IR_INCLUDE ../ir/include ABSOLUTE)
add_subdirectory(include)
add_subdirectory(src)

set(src src/AsseraPasses.cpp)

set(rcvalue_src
    src/value/BarrierOptPass.cpp
    src/value/FunctionPointerResolutionPass.cpp
    src/value/RangeValueOptimizePass.cpp
    src/value/ValueFuncToTargetPass.cpp
    src/value/ValueUnrollingPass.cpp
    src/value/ValueSimplifyPass.cpp
    src/value/ValueToLLVMLoweringPass.cpp
    src/value/ValueToStandardLoweringPass.cpp
)

set(rcvalue_include
    include/value/BarrierOptPass.h
    include/value/FunctionPointerResolutionPass.h
    include/value/RangeValueOptimizePass.h
    include/value/ValueFuncToTargetPass.h
    include/value/ValueUnrollingPass.h
    include/value/ValueSimplifyPass.h
    include/value/ValueToLLVMLoweringPass.h
    include/value/ValueToStandardLoweringPass.h
)

set(rcnest_src
  src/nest/LoopNestPasses.cpp
  src/nest/LoopNestToValue.cpp
  src/nest/LoopNestToValueFunc.cpp
)

set(include include/AsseraPasses.h)

set(rcnest_include
  include/nest/LoopNestPasses.h
  include/nest/LoopNestToValue.h
  include/nest/LoopNestToValueFunc.h
)

set(rcexec_src src/exec/ExecutionPlanToAffineLoweringPass.cpp)

set(rcexec_include include/exec/ExecutionPlanToAffineLoweringPass.h)

set(rcgpu_src
  src/gpu/AsseraToGPUPass.cpp
  src/gpu/ConvertLaunchFuncToVulkanCalls.cpp
  src/gpu/EmitVulkanWrappers.cpp
  src/gpu/SerializeToHSACO.cpp
)

set(rcgpu_include
  include/gpu/AsseraToGPUPass.h
  include/gpu/AsseraVulkanPasses.h
)

set(accaffine_src
  src/affine/AffineLoopNormalize.cpp
  src/affine/AffineSimplifications.cpp
  src/affine/CheckBoundsPass.cpp
)

set(accaffine_include
  include/affine/AffineLoopNormalize.h
  include/affine/AffineSimplifications.h
  include/affine/CheckBoundsPass.h
)

set(accvec_src
  src/vectorization/VectorizationPass.cpp
  src/vectorization/VectorizationUtil.cpp
  src/vectorization/VectorizedOp.cpp
)

set(accvec_include
  include/vectorization/VectorizationPass.h
  include/vectorization/VectorizationUtil.h
  include/vectorization/VectorizedOp.h
)

set(util_src
  src/util/DebugFunctionPass.cpp
  src/util/MathUtilities.cpp
  src/util/RangeValueUtilities.cpp
  src/util/SnapshotUtilities.cpp

  # Disabled
  # src/util/DebugUtilities.cpp
)

set(util_include
  include/util/DebugFunctionPass.h
  include/util/MathUtilities.h
  include/util/RangeValueUtilities.h
  include/util/SnapshotUtilities.h

  # Disabled
  # include/util/DebugUtilities.h
)


set(library_name transforms)
set(src
    ${src}
    ${rcvalue_src}
    ${rcnest_src}
    ${rcexec_src}
    ${rcgpu_src}
    ${accaffine_src}
    ${accvec_src}
    ${util_src}
)

set(include
    ${include}
    ${rcvalue_include}
    ${rcnest_include}
    ${rcexec_include}
    ${rcgpu_include}
    ${accaffine_include}
    ${accvec_include}
    ${util_include}
)

add_library(${library_name} ${src} ${include})
target_include_directories(
  ${library_name} PRIVATE include ${ASSERA_LIBRARIES_DIR}
   PUBLIC
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
   $<INSTALL_INTERFACE:include/assera/transforms/include>
   $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src>
   $<INSTALL_INTERFACE:include/assera/transforms/src>
)
target_include_directories(${library_name} SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_link_libraries(
  ${library_name}
  PUBLIC utilities
         ir
         value
         lldELF
         MLIRGPUOps
         MLIRROCDLIR
         MLIRROCDLToLLVMIRTranslation
         MLIRStandardToLLVM
         MLIRSCFToControlFlow
         MLIRControlFlowToLLVM
         MLIRAffineToStandard
         MLIRAffineTransforms
         MLIRAffineUtils
         MLIRLinalgToLLVM
         MLIRLinalgTransforms
         MLIRTargetLLVMIRExport
         MLIRSupport
         MLIRIR
         MLIRAnalysis
         MLIRLinalg
         MLIRLLVMIR
         MLIRParser
         MLIRPass
         MLIRTransforms
)

add_dependencies(
  ${library_name}
  ValueConversionIncGen
  LoopNestToValueIncGen
  AsseraConversionIncGen
)

#
# Install headers and library
#
InstallAsseraHeaders(
  INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
               ${CMAKE_CURRENT_LIST_DIR}/include
               ${CMAKE_CURRENT_BINARY_DIR}/src
)
InstallAsseraLibrary(${library_name})

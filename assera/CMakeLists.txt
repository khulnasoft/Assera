####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

set(ASSERA_LIBRARIES_DIR ${CMAKE_CURRENT_LIST_DIR})
set(ASSERA_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ASSERA_LIBRARIES_DIR})

add_subdirectory(acc-opt)
add_subdirectory(acc-gpu-runner)
add_subdirectory(acc-lsp-server)
add_subdirectory(acc-translate)
add_subdirectory(accc)
add_subdirectory(ir)
add_subdirectory(mlirHelpers)
add_subdirectory(onnx-emitter)
add_subdirectory(runtime)
add_subdirectory(testing)
add_subdirectory(toml)
add_subdirectory(transforms)
add_subdirectory(utilities)
add_subdirectory(value)
add_subdirectory(vulkanRuntime)
add_subdirectory(export)

add_custom_target(assera)
add_dependencies(
  assera
  ir
  mlirHelpers
  runtime
  transforms
  utilities
  value
)

add_custom_target(tests)
add_dependencies(
  tests
  nest_dialect_test
  testing
  utilities_test
)

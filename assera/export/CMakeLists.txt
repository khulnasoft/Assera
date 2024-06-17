####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################
if(NOT "${LLVM_SETUP_VARIANT}" STREQUAL "Conan")
  return()
endif()

# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(ASSERA_INSTALL_PACKAGE_DIR lib/cmake/assera)
set(ASSERA_CMAKE_BUILDDIR "${CMAKE_BINARY_DIR}/${ASSERA_INSTALL_PACKAGE_DIR}")

get_property(ASSERA_EXPORTED_LIBS GLOBAL PROPERTY ASSERA_EXPORTED_LIBS)
get_property(ASSERA_RUNTIME_LIBS GLOBAL PROPERTY ASSERA_RUNTIME_LIBS)
set(ALL_EXPORTED_TARGETS ${ASSERA_EXPORTED_LIBS} ${ASSERA_RUNTIME_LIBS})
export(TARGETS ${ALL_EXPORTED_TARGETS} FILE ${ASSERA_CMAKE_BUILDDIR}/AsseraTargets.cmake)

# Set variables for text-replacement in AsseraConfig.cmake.in -> AsseraConfig.cmake

# Generate AsseraConfig.cmake for the build tree.
set(ASSERA_CONFIG_CMAKE_DIR "${ASSERA_CMAKE_BUILDDIR}")
set(ASSERA_CONFIG_EXPORTS_FILE "\${ASSERA_CMAKE_DIR}/AsseraTargets.cmake") # AsseraConfig.cmake must set ASSERA_CMAKE_DIR before including ASSERA_CONFIG_EXPORTS_FILE
set(ASSERA_CONFIG_INCLUDE_DIRS
  "${CMAKE_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AsseraConfig.cmake.in
  ${ASSERA_CMAKE_BUILDDIR}/AsseraConfig.cmake
  @ONLY)
set(ASSERA_CONFIG_CMAKE_DIR)
set(ASSERA_CONFIG_EXPORTS_FILE)
set(ASSERA_CONFIG_INCLUDE_DIRS)

# Generate AsseraConfig.cmake for the install tree.
set(ASSERA_CONFIG_CODE "
# Compute the installation prefix from this AsseraConfig.cmake file location.
get_filename_component(ASSERA_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${ASSERA_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(ASSERA_CONFIG_CODE "${ASSERA_CONFIG_CODE}
get_filename_component(ASSERA_INSTALL_PREFIX \"\${ASSERA_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(ASSERA_CONFIG_CMAKE_DIR "\${ASSERA_INSTALL_PREFIX}/${ASSERA_INSTALL_PACKAGE_DIR}")
set(ASSERA_CONFIG_EXPORTS_FILE "\${ASSERA_CMAKE_DIR}/AsseraTargets.cmake")
set(ASSERA_CONFIG_INCLUDE_DIRS
  "\${ASSERA_INSTALL_PREFIX}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AsseraConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AsseraConfig.cmake
  @ONLY)
set(ASSERA_CONFIG_CODE)
set(ASSERA_CONFIG_CMAKE_DIR)
set(ASSERA_CONFIG_EXPORTS_FILE)
set(ASSERA_CONFIG_INCLUDE_DIRS)

install(EXPORT AsseraTargets DESTINATION ${ASSERA_INSTALL_PACKAGE_DIR}
        COMPONENT assera-cmake-exports)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AsseraConfig.cmake
  DESTINATION ${ASSERA_INSTALL_PACKAGE_DIR}
  COMPONENT assera-cmake-exports)

# Install cmake files to enable generator projects to get LLVM/MLIR, etc
install(FILES
  ${CMAKE_SOURCE_DIR}/CMake/LLVMSetup.cmake
  ${CMAKE_SOURCE_DIR}/CMake/LLVMSetup${LLVM_SETUP_VARIANT}.cmake
  ${CMAKE_SOURCE_DIR}/CMake/MKLSetup.cmake
  ${CMAKE_SOURCE_DIR}/CMake/OpenBLASSetup.cmake
  DESTINATION ${ASSERA_INSTALL_PACKAGE_DIR}
  COMPONENT assera-cmake-exports)

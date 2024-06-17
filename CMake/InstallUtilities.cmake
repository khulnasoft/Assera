####################################################################################################
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for license information.
####################################################################################################

# Macro to install headers while keeping directory structure
macro(InstallAsseraHeaders)
  set(options )
  set(oneValueArgs )
  set(multiValueArgs INCLUDE_DIRS)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  file(RELATIVE_PATH current_relative_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_LIST_DIR})

  foreach(include_dir ${ARG_INCLUDE_DIRS})
    install(DIRECTORY ${include_dir}
            DESTINATION include/${current_relative_path}
            COMPONENT assera-headers
            FILES_MATCHING
              PATTERN "*.def"
              PATTERN "*.h"
              PATTERN "*.gen"
              PATTERN "*.inc"
              PATTERN "*.td"
              PATTERN "CMakeFiles" EXCLUDE
              PATTERN "config.h" EXCLUDE
              PATTERN "*.tlog" EXCLUDE
            )
  endforeach()
endmacro()

# Macros to install libraries
function(InstallAsseraLibrary library_name)
  install(TARGETS ${library_name}
          EXPORT AsseraTargets
          LIBRARY)
  set_property(GLOBAL APPEND PROPERTY ASSERA_EXPORTED_LIBS ${library_name})
endfunction()

function(InstallAsseraCppLibrary library_name)
  install(TARGETS ${library_name}
          EXPORT AsseraTargets
          LIBRARY)
  set_property(GLOBAL APPEND PROPERTY ASSERA_RUNTIME_LIBS ${library_name})
endfunction()

function(InstallAsseraPyLibrary library_name component destination)
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER ${CONFIG} CONFIG_UPPER)
      install(TARGETS ${library_name}
              COMPONENT ${component}
              CONFIGURATIONS ${CONFIG}
              ARCHIVE DESTINATION ${destination}
              RUNTIME DESTINATION ${destination}
              LIBRARY DESTINATION ${destination}
      )
  endforeach(CONFIG ${CMAKE_CONFIGURATION_TYPES})

  install(TARGETS ${library_name}
          COMPONENT ${component}
          ARCHIVE DESTINATION ${destination}
          RUNTIME DESTINATION ${destination}
          LIBRARY DESTINATION ${destination}
  )
endfunction()

# Macros to install runtimes (binaries)
function(InstallAsseraRuntime runtime_name)
  install(TARGETS ${runtime_name} RUNTIME)
endfunction()

function(InstallAsseraPyRuntimeLibrary library_name component destination)
  InstallAsseraRuntime(${library_name})
  InstallAsseraPyLibrary(${library_name} ${component} ${destination})
endfunction()

function(InstallAsseraDirectory directory component destination)
  install(DIRECTORY ${directory}/ # trailing slash is important!
          COMPONENT ${component}
          DESTINATION ${destination}
  )
endfunction()
# ==================== SceneTree Common Infrastructure ====================
# Shared JSON-RPC 2.0 infrastructure for SceneTree Passthrough.
#
# This sublibrary provides:
#   - Core JSON-RPC server framework (jsonrpc_server.hpp/cpp)
#   - JSON building utilities (handler_base.hpp)
#   - Handler dispatch infrastructure
#
# Handler implementations are in subdirectories:
#   - handlers/real/     - Real SceneTree API (for godot-sandbox embedding)
#   - handlers/mock/     - Mock implementations (for standalone/stdio testing)
#
# Usage:
#   add_subdirectory(path/to/scenetree_common)
#   target_link_libraries(my_target PRIVATE scenetree_common)
#
# To select handler implementation, define before add_subdirectory:
#   set(SCENETREE_HANDLERS "real"  CACHE STRING "Handler implementation: real or mock")
#
# ============================================================================

cmake_minimum_required(VERSION 3.15)
project(scenetree_common)

# ==================== Headers ====================
set(COMMON_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/handler_base.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jsonrpc_server.hpp"
)

# ==================== Sources ====================
set(COMMON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/jsonrpc_server.cpp"
)

# ==================== Library ====================
add_library(scenetree_common STATIC ${COMMON_SOURCES} ${COMMON_HEADERS})
target_include_directories(scenetree_common PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# C interface
target_compile_features(scenetree_common PUBLIC c_std_99)

# Compiler warnings
target_compile_options(scenetree_common PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

# ==================== Handler Implementation ====================
# Choose handler implementation based on SCENETREE_HANDLERS setting
#
# Options:
#   - "real"   - Real SceneTree/Node API handlers (uses Godot C++ API)
#   - "mock"   - Mock handlers for testing/stdio mode (no Godot dependency)
#   - "none"   - No handlers (frontend provides its own)

set(SCENETREE_HANDLERS "${SCENETREE_HANDLERS}" CACHE STRING "Handler implementation: real, mock, or none")
message(STATUS "SceneTree Common: Handler implementation = ${SCENETREE_HANDLERS}")

if(SCENETREE_HANDLERS STREQUAL "real")
    message(STATUS "Using REAL handlers (embedded godot-sandbox mode)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/handlers/real ${CMAKE_BINARY_DIR}/handlers_real)
    target_link_libraries(scenetree_common PRIVATE handlers_real)

elseif(SCENETREE_HANDLERS STREQUAL "mock")
    message(STATUS "Using MOCK handlers (standalone/stdio testing mode)")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/handlers/mock ${CMAKE_BINARY_DIR}/handlers_mock)
    target_link_libraries(scenetree_common PRIVATE handlers_mock)

elseif(SCENETREE_HANDLERS STREQUAL "none")
    message(STATUS "NOT building SceneTree handlers (frontend will provide its own)")

else()
    message(WARNING "Unknown SCENETREE_HANDLERS value: ${SCENETREE_HANDLERS}. Using 'none'.")
    set(SCENETREE_HANDLERS "none")
endif()

# ==================== Install ====================
install(TARGETS scenetree_common
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/scenetree_common
)

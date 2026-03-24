# ==================== Mock Handlers ====================
# Mock implementations for testing without Godot API access.
# These are simple pass-through handlers that return mock responses.

cmake_minimum_required(VERSION 3.15)
project(handlers_mock)

set(HANDLER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/mock_handlers.cpp"
)

add_library(handlers_mock STATIC ${HANDLER_SOURCES})
target_include_directories(handlers_mock PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
)

target_compile_options(handlers_mock PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

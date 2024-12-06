cmake_minimum_required(VERSION 3.10)
project(automerge)

find_package(Cargo REQUIRED)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TARGET_DIR ${PROJECT_ROOT_DIR}/target/release)

add_custom_target(automerge_rust ALL
	COMMAND ${CARGO_EXECUTABLE} build --release
	WORKING_DIRECTORY ${PROJECT_ROOT_DIR}
	BYPRODUCTS ${TARGET_DIR}/automerge
)

add_executable(automerge_rust_exec IMPORTED)
set_target_properties(automerge_rust_exec PROPERTIES
	IMPORTED_LOCATION ${TARGET_DIR}/automerge
)

add_dependencies(automerge_rust_exec automerge_rust)
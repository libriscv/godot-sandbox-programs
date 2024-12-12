cmake_minimum_required(VERSION 3.10)
project(cjit)

include(FetchContent)
FetchContent_Declare(tinycc
	GIT_REPOSITORY https://github.com/TinyCC/tinycc.git
	GIT_TAG        mob
)
FetchContent_MakeAvailable(tinycc)

set(CONFIG_H_LOCATION "${CMAKE_BINARY_DIR}/_deps/tinycc-src/config.h")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CONFIG_H_LOCATION})
set_source_files_properties(${CONFIG_H_LOCATION} PROPERTIES GENERATED TRUE)
file(COPY "${CMAKE_BINARY_DIR}/_deps/tinycc-src/libtcc.h" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(libtcc STATIC
	${tinycc_SOURCE_DIR}/libtcc.c
	${tinycc_SOURCE_DIR}/libtcc.h
	${CONFIG_H_LOCATION}
)
target_compile_definitions(libtcc PRIVATE
	CONFIG_TCC_PREDEFS=1
	TCC_TARGET_RISCV64=1
	TARGETOS_Linux=1
	TCC_VERSION="0.9.27"
)
target_include_directories(libtcc PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}"
)


add_ci_program(cjit
	main.cpp
)
target_link_libraries(cjit PRIVATE libtcc)

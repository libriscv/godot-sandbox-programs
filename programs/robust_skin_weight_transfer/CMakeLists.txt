cmake_minimum_required(VERSION 3.14)
project(robust_weight_transfer)

add_ci_program(robust_weight_transfer
	robust_weight_transfer.cpp
	thirdparty/nonmanifold-laplacian/src/bubble_offset.cpp
	thirdparty/nonmanifold-laplacian/src/point_cloud_utilities.cpp
	thirdparty/nonmanifold-laplacian/src/utils.hpp
	thirdparty/nonmanifold-laplacian/src/process_mesh.cpp
)

target_include_directories(robust_weight_transfer SYSTEM PRIVATE 
	thirdparty/nonmanifold-laplacian/include 
	thirdparty/nonmanifold-laplacian/include 
)

target_link_libraries(robust_weight_transfer PRIVATE eigen geometry-central igl::core)
target_compile_definitions(robust_weight_transfer PRIVATE
	-D_USE_MATH_DEFINES
	-DIGL_PARALLEL_FOR_FORCE_SERIAL
)

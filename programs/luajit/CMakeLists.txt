cmake_minimum_required(VERSION 3.10)
project(luajit)

add_ci_program(luajit
	main.cpp
)
target_link_libraries(luajit PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/libluajit.a
)
target_include_directories(luajit PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_SOURCE_DIR}/ext/LuaJIT/src
)

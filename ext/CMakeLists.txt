
if (NOT EMBEDDED_RISCV)
	# Eigen C++ linear algebra library
	add_subdirectory(eigen)

	# libigl C++ geometry processing library
	add_subdirectory(libigl)

	# geometry-central C++ geometry processing library
	add_subdirectory(geometry-central)
endif()

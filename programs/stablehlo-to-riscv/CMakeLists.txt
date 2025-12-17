cmake_minimum_required(VERSION 3.13.4)
project(stablehlo-to-riscv)

include(ExternalProject)
include(FindPkgConfig)

# Try to find Homebrew's LLVM/MLIR/StableHLO installation first
set(HOMEBREW_PREFIX "/opt/homebrew")
if(APPLE AND EXISTS "${HOMEBREW_PREFIX}")
    set(CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}" ${CMAKE_PREFIX_PATH})
endif()

# Try to find MLIR via CMake
find_package(MLIR QUIET)
find_package(LLVM QUIET)

# Try to find StableHLO - check multiple possible library names and paths
find_path(STABLEHLO_INCLUDE_DIR
    NAMES stablehlo/dialect/StablehloOps.h
    PATHS
        ${HOMEBREW_PREFIX}/include
        /usr/local/include
        /usr/include
)

# Try to find StableHLO libraries (may have different names, including static libs)
# Check for both shared and static libraries
find_library(STABLEHLO_DIALECT_LIB
    NAMES StablehloDialect libStablehloDialect libMLIRStablehloDialect.a MLIRStablehloDialect
    PATHS
        ${HOMEBREW_PREFIX}/lib
        /usr/local/lib
        /usr/lib
)

find_library(STABLEHLO_TRANSFORMS_LIB
    NAMES StablehloTransforms libStablehloTransforms libMLIRStablehloTransforms.a MLIRStablehloTransforms
    PATHS
        ${HOMEBREW_PREFIX}/lib
        /usr/local/lib
        /usr/lib
)

# Also check for static libraries explicitly
find_file(STABLEHLO_DIALECT_STATIC
    NAMES libMLIRStablehloDialect.a libStablehloDialect.a
    PATHS
        ${HOMEBREW_PREFIX}/lib
        /usr/local/lib
        /usr/lib
)

find_file(STABLEHLO_TRANSFORMS_STATIC
    NAMES libMLIRStablehloTransforms.a libStablehloTransforms.a
    PATHS
        ${HOMEBREW_PREFIX}/lib
        /usr/local/lib
        /usr/lib
)

# Set STABLEHLO_LIBRARY if found (prefer static if available)
if(STABLEHLO_DIALECT_STATIC AND STABLEHLO_TRANSFORMS_STATIC)
    set(STABLEHLO_LIBRARY ${STABLEHLO_DIALECT_STATIC} ${STABLEHLO_TRANSFORMS_STATIC})
elseif(STABLEHLO_DIALECT_LIB AND STABLEHLO_TRANSFORMS_LIB)
    set(STABLEHLO_LIBRARY ${STABLEHLO_DIALECT_LIB} ${STABLEHLO_TRANSFORMS_LIB})
elseif(STABLEHLO_DIALECT_LIB)
    set(STABLEHLO_LIBRARY ${STABLEHLO_DIALECT_LIB})
elseif(STABLEHLO_TRANSFORMS_LIB)
    set(STABLEHLO_LIBRARY ${STABLEHLO_TRANSFORMS_LIB})
endif()

set(USE_HOMEBREW_LLVM FALSE)
if(MLIR_FOUND AND LLVM_FOUND AND STABLEHLO_INCLUDE_DIR AND STABLEHLO_LIBRARY)
    message(STATUS "Using Homebrew's LLVM/MLIR/StableHLO installation")
    set(USE_HOMEBREW_LLVM TRUE)
    set(LLVM_DIR ${LLVM_DIR})
    set(MLIR_DIR ${MLIR_DIR})
    set(STABLEHLO_INSTALL_PREFIX ${HOMEBREW_PREFIX})
else()
    message(STATUS "Homebrew LLVM/MLIR/StableHLO not found, will build as ExternalProjects")
    
    # Set up directories for external projects
    set(EXTERNAL_PROJECTS_DIR ${CMAKE_BINARY_DIR}/external)
    set(LLVM_INSTALL_PREFIX ${EXTERNAL_PROJECTS_DIR}/llvm-install)
    set(STABLEHLO_INSTALL_PREFIX ${EXTERNAL_PROJECTS_DIR}/stablehlo-install)
    
    # Build LLVM with MLIR enabled using stablehlo's tested commit
    # Use the LLVM commit hash from stablehlo project for better compatibility
    ExternalProject_Add(
        llvm-project
        GIT_REPOSITORY https://github.com/llvm/llvm-project.git
        GIT_TAG 0c2701fe7fa002e1befc5f86c268a7964f96d286
        GIT_SHALLOW TRUE
        SOURCE_DIR ${EXTERNAL_PROJECTS_DIR}/llvm-project-src
        SOURCE_SUBDIR llvm
        BINARY_DIR ${EXTERNAL_PROJECTS_DIR}/llvm-project-build
        INSTALL_DIR ${LLVM_INSTALL_PREFIX}
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
            -DCMAKE_BUILD_TYPE=Release
            -DLLVM_ENABLE_PROJECTS=mlir
            -DLLVM_TARGETS_TO_BUILD=AArch64;RISCV
            -DLLVM_ENABLE_ASSERTIONS=OFF
            -DLLVM_ENABLE_BINDINGS=OFF
            -DLLVM_ENABLE_OCAMLDOC=OFF
            -DLLVM_ENABLE_ZLIB=ON
            -DLLVM_INCLUDE_EXAMPLES=OFF
            -DLLVM_INCLUDE_TESTS=OFF
            -DLLVM_INCLUDE_DOCS=OFF
            -DLLVM_BUILD_EXAMPLES=OFF
            -DLLVM_BUILD_TESTS=OFF
            -DLLVM_BUILD_DOCS=OFF
            -DLLVM_ENABLE_RTTI=ON
            -DLLVM_ENABLE_EH=ON
            -DLLVM_ENABLE_TERMINFO=OFF
            -DLLVM_ENABLE_LIBEDIT=OFF
            -DLLVM_ENABLE_LIBXML2=OFF
            -DLLVM_ENABLE_ZSTD=OFF
            -DCMAKE_BUILD_RPATH=${LLVM_INSTALL_PREFIX}/lib
            -DCMAKE_INSTALL_RPATH=${LLVM_INSTALL_PREFIX}/lib
            -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
            -DMLIR_INCLUDE_TESTS=OFF
            -DLLVM_VERSION_SUFFIX=""
            -DCMAKE_PLATFORM_NO_VERSIONED_SONAME:BOOL=ON
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install .
        UPDATE_COMMAND ""
    )
        
    # Build StableHLO (depends on LLVM/MLIR)
    ExternalProject_Add(
        stablehlo
        GIT_REPOSITORY https://github.com/openxla/stablehlo.git
        GIT_TAG v1.0.0
        GIT_SHALLOW TRUE
        SOURCE_DIR ${EXTERNAL_PROJECTS_DIR}/stablehlo-src
        BINARY_DIR ${EXTERNAL_PROJECTS_DIR}/stablehlo-build
        INSTALL_DIR ${STABLEHLO_INSTALL_PREFIX}
        DEPENDS llvm-project
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${STABLEHLO_INSTALL_PREFIX}
            -DCMAKE_BUILD_TYPE=Release
            -DMLIR_DIR=${LLVM_INSTALL_PREFIX}/lib/cmake/mlir
            -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF
            -DSTABLEHLO_ENABLE_TESTS=OFF
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        BUILD_COMMAND ${CMAKE_COMMAND} --build . --parallel ${CMAKE_BUILD_PARALLEL_LEVEL}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install .
        UPDATE_COMMAND ""
    )
    
    set(LLVM_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/llvm)
    set(MLIR_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/mlir)
endif()

# Set up include directories
if(USE_HOMEBREW_LLVM)
    # Use Homebrew paths
    include_directories(
        ${LLVM_INCLUDE_DIRS}
        ${STABLEHLO_INCLUDE_DIR}
    )
else()
    # Use ExternalProject paths
    include_directories(
        ${LLVM_INSTALL_PREFIX}/include
        ${STABLEHLO_INSTALL_PREFIX}/include
    )
endif()

# Build the stablehlo-to-riscv executable (for host, not RISC-V)
add_ci_program(stablehlo_to_riscv
    main.cpp
    mlir_compiler.cpp
)

# Remove RISC-V flags for host builds (clang++ doesn't support RISC-V arch flags on macOS)
# These flags are added by godot-sandbox but we're building for host, not cross-compiling
# Function to remove RISC-V flags from a target
function(remove_riscv_flags target_name)
    if(TARGET ${target_name})
        get_target_property(OPTS ${target_name} COMPILE_OPTIONS)
        if(OPTS)
            list(FILTER OPTS EXCLUDE REGEX "^-march=.*")
            list(FILTER OPTS EXCLUDE REGEX "^-mabi=.*")
            set_target_properties(${target_name} PROPERTIES COMPILE_OPTIONS "${OPTS}")
        endif()
    endif()
endfunction()

# Remove flags from sandbox_api (needed for stablehlo-to-riscv to link properly)
if(TARGET sandbox_api)
    remove_riscv_flags(sandbox_api)
endif()

# Remove flags from stablehlo_to_riscv target
remove_riscv_flags(stablehlo_to_riscv)

# Also override the variables to prevent them from being used in future targets
set(RISCV_ARCH "" CACHE STRING "" FORCE)
set(RISCV_ABI "" CACHE STRING "" FORCE)

# Link directories
if(USE_HOMEBREW_LLVM)
    target_link_directories(stablehlo_to_riscv PRIVATE
        ${LLVM_LIBRARY_DIRS}
        ${HOMEBREW_PREFIX}/lib
    )
else()
    target_link_directories(stablehlo_to_riscv PRIVATE
        ${LLVM_INSTALL_PREFIX}/lib
        ${STABLEHLO_INSTALL_PREFIX}/lib
    )
endif()

# Link against MLIR libraries
if(USE_HOMEBREW_LLVM)
    # Use CMake targets from Homebrew's LLVM/MLIR
    target_link_libraries(stablehlo_to_riscv PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRConversion
        MLIRTargetLLVMIR
        MLIRExecutionEngine
        MLIRFuncDialect
        MLIRArithDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRVectorDialect
        MLIRTensorDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRLLVMDialect
        MLIRLinalgDialect
        MLIRAffineDialect
        MLIRStandardDialect
        # StableHLO from Homebrew
        ${STABLEHLO_LIBRARY}
        # LLVM libraries
        LLVMCore
        LLVMSupport
        LLVMTarget
        LLVMRISCVCodeGen
        LLVMMC
        LLVMObject
        LLVMAsmParser
        LLVMBitWriter
        LLVMIRReader
        LLVMLinker
        LLVMExecutionEngine
        LLVMOption
        LLVMDemangle
    )
else()
    # Link against ExternalProject libraries
    target_link_libraries(stablehlo_to_riscv PRIVATE
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRConversion
        MLIRTargetLLVMIR
        MLIRExecutionEngine
        MLIRFuncDialect
        MLIRArithDialect
        MLIRMathDialect
        MLIRMemRefDialect
        MLIRVectorDialect
        MLIRTensorDialect
        MLIRSCFDialect
        MLIRControlFlowDialect
        MLIRLLVMDialect
        MLIRLinalgDialect
        MLIRAffineDialect
        MLIRStandardDialect
        MLIRStablehloDialect
        MLIRStablehloTransforms
        # LLVM libraries
        LLVMCore
        LLVMSupport
        LLVMTarget
        LLVMRISCVCodeGen
        LLVMMC
        LLVMObject
        LLVMAsmParser
        LLVMBitWriter
        LLVMIRReader
        LLVMLinker
        LLVMExecutionEngine
        LLVMOption
        LLVMDemangle
    )
    # Ensure external projects are built before our target
    add_dependencies(stablehlo_to_riscv llvm-project stablehlo)
endif()

# Compile definitions
target_compile_definitions(stablehlo_to_riscv PRIVATE USE_MLIR=1)

# Add LLVM/MLIR compile flags manually
# Note: LLVM is built with RTTI enabled, so we match that
target_compile_options(stablehlo_to_riscv PRIVATE
    -fexceptions
    -frtti
)

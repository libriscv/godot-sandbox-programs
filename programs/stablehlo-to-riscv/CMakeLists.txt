cmake_minimum_required(VERSION 3.13.4)
project(stablehlo-to-riscv)

include(ExternalProject)

# Set up directories for external projects
set(EXTERNAL_PROJECTS_DIR ${CMAKE_BINARY_DIR}/external)
set(LLVM_INSTALL_PREFIX ${EXTERNAL_PROJECTS_DIR}/llvm-install)
set(STABLEHLO_INSTALL_PREFIX ${EXTERNAL_PROJECTS_DIR}/stablehlo-install)

# Build LLVM with MLIR enabled
ExternalProject_Add(
    llvm-project
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG llvmorg-18.1.0
    GIT_SHALLOW TRUE
    SOURCE_DIR ${EXTERNAL_PROJECTS_DIR}/llvm-project-src
    BINARY_DIR ${EXTERNAL_PROJECTS_DIR}/llvm-project-build
    INSTALL_DIR ${LLVM_INSTALL_PREFIX}
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=Release
        -DLLVM_ENABLE_PROJECTS=mlir
        -DLLVM_TARGETS_TO_BUILD="X86;RISCV"
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
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
        -DMLIR_INCLUDE_TESTS=OFF
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

# Set paths (will exist after external projects build)
set(LLVM_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_INSTALL_PREFIX}/lib/cmake/mlir)

# Include directories (manually set since we know where they'll be)
include_directories(
    ${LLVM_INSTALL_PREFIX}/include
    ${STABLEHLO_INSTALL_PREFIX}/include
)

# Build the stablehlo-to-riscv RISC-V executable
add_ci_program(stablehlo_to_riscv
    main.cpp
    mlir_compiler.cpp
)

# Remove RISC-V specific flags that clang++ doesn't support on macOS
# These flags are added by add_sandbox_program but we're building for host, not RISC-V
# We need to filter them out from compile options
get_target_property(OLD_COMPILE_OPTIONS stablehlo_to_riscv COMPILE_OPTIONS)
if(OLD_COMPILE_OPTIONS)
    set(NEW_COMPILE_OPTIONS "")
    foreach(OPT ${OLD_COMPILE_OPTIONS})
        if(NOT OPT MATCHES "^-march=.*" AND NOT OPT MATCHES "^-mabi=.*")
            list(APPEND NEW_COMPILE_OPTIONS ${OPT})
        endif()
    endforeach()
    set_target_properties(stablehlo_to_riscv PROPERTIES COMPILE_OPTIONS "${NEW_COMPILE_OPTIONS}")
endif()

# Link directories
target_link_directories(stablehlo_to_riscv PRIVATE
    ${LLVM_INSTALL_PREFIX}/lib
    ${STABLEHLO_INSTALL_PREFIX}/lib
)

# Link against MLIR libraries (we'll link them by name since they'll be built)
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

# Compile definitions
target_compile_definitions(stablehlo_to_riscv PRIVATE USE_MLIR=1)

# Add LLVM/MLIR compile flags manually
# Note: LLVM is built with RTTI enabled, so we match that
target_compile_options(stablehlo_to_riscv PRIVATE
    -fexceptions
    -frtti
)

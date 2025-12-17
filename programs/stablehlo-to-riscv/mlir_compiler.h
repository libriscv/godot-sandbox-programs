#pragma once

#include <string>
#include <vector>

class MLIRCompiler {
public:
    // Compile StableHLO MLIR text directly to RISC-V ELF64 binary
    // Returns binary data, or empty vector on error
    static std::vector<uint8_t> compile_stablehlo_to_riscv_elf64(const std::string &stablehlo_mlir);
    
    // Check if MLIR is available
    static bool is_mlir_available();
    
    // Get MLIR version info
    static std::string get_mlir_version();

private:
    // Lower StableHLO through MLIR dialects to LLVM IR
    static bool lower_to_llvm(const std::string &stablehlo_mlir, std::string &llvm_ir);
    
    // Compile LLVM IR to RISC-V ELF64
    static std::vector<uint8_t> compile_llvm_to_riscv_elf64(const std::string &llvm_ir);
};

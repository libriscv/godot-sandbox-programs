#include <api.hpp>
#include "mlir_compiler.h"
#include <string>

// Check if MLIR is available
static Variant is_mlir_available() {
    return MLIRCompiler::is_mlir_available();
}

// Get MLIR version
static Variant get_mlir_version() {
    return MLIRCompiler::get_mlir_version().c_str();
}

// Compile StableHLO directly to RISC-V ELF64 binary
static Variant compile_stablehlo_to_riscv_elf64(String stablehlo_content) {
    std::string content = stablehlo_content.utf8().get_data();
    
    print("Compiling StableHLO to RISC-V ELF64 using MLIR");
    std::vector<uint8_t> elf64_binary = MLIRCompiler::compile_stablehlo_to_riscv_elf64(content);
    
    if (elf64_binary.empty()) {
        return "Error: Failed to compile StableHLO to RISC-V ELF64. Check MLIR setup.";
    }
    
    // Convert to PackedByteArray
    PackedByteArray result;
    result.resize(elf64_binary.size());
    for (size_t i = 0; i < elf64_binary.size(); i++) {
        result[i] = elf64_binary[i];
    }
    
    return result;
}

int main() {
    print("StableHLO to RISC-V compiler initialized (MLIR-based)");
    
    // The entire Godot API is available
    Sandbox sandbox = get_node<Sandbox>();
    print(sandbox.is_binary_translated()
        ? "The current program is accelerated by binary translation."
        : "The current program is running in interpreter mode.");
    
    // Add public API functions
    ADD_API_FUNCTION(compile_stablehlo_to_riscv_elf64, "PackedByteArray", "String stablehlo_content", 
                     "Compiles StableHLO content directly to RISC-V ELF64 binary using MLIR");
    ADD_API_FUNCTION(is_mlir_available, "bool", "", 
                     "Checks if MLIR compiler is available");
    ADD_API_FUNCTION(get_mlir_version, "String", "", 
                     "Returns MLIR version information");
    
    halt();
}

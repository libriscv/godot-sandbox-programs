#include <api.hpp>
#include "stablehlo_parser.h"
#include "cpp_generator.h"
#include "riscv_compiler.h"
#include <string>

// Parse StableHLO from string and return parsed function info
static Variant parse_stablehlo(String stablehlo_content) {
    std::string content = stablehlo_content.utf8().get_data();
    StableHLOFunction func = StableHLOParser::parse_from_string(content);
    
    if (func.name.empty()) {
        return "Error: Failed to parse StableHLO content";
    }
    
    // Validate operations
    int unsupported_count = 0;
    for (const auto &op : func.operations) {
        if (!StableHLOParser::is_supported_operation(op.op_name)) {
            unsupported_count++;
            print("Warning: Unsupported operation: ", op.op_name.c_str());
        }
    }
    
    Dictionary result;
    result["name"] = func.name.c_str();
    result["operations_count"] = func.operations.size();
    result["unsupported_operations"] = unsupported_count;
    result["arguments_count"] = func.arguments.size();
    
    return result;
}

// Generate C++ code from StableHLO string
static Variant generate_cpp_from_stablehlo(String stablehlo_content) {
    std::string content = stablehlo_content.utf8().get_data();
    StableHLOFunction func = StableHLOParser::parse_from_string(content);
    
    if (func.name.empty()) {
        return "Error: Failed to parse StableHLO content";
    }
    
    // Generate C++ code
    std::string cpp_code = CPPGenerator::generate_cpp(func);
    
    return cpp_code.c_str();
}

// Check if RISC-V compiler is available
static Variant is_riscv_compiler_available() {
    return RISCVCompiler::is_compiler_available();
}

// Get RISC-V compiler path
static Variant get_riscv_compiler_path() {
    std::string path = RISCVCompiler::get_compiler_path();
    if (path.empty()) {
        return "No RISC-V compiler found";
    }
    return path.c_str();
}

int main() {
    print("StableHLO to RISC-V compiler initialized");
    
    // The entire Godot API is available
    Sandbox sandbox = get_node<Sandbox>();
    print(sandbox.is_binary_translated()
        ? "The current program is accelerated by binary translation."
        : "The current program is running in interpreter mode.");
    
    // Add public API functions
    ADD_API_FUNCTION(parse_stablehlo, "Dictionary", "String stablehlo_content", 
                     "Parses StableHLO content and returns function information");
    ADD_API_FUNCTION(generate_cpp_from_stablehlo, "String", "String stablehlo_content", 
                     "Generates C++ code from StableHLO content");
    ADD_API_FUNCTION(is_riscv_compiler_available, "bool", "", 
                     "Checks if RISC-V compiler is available");
    ADD_API_FUNCTION(get_riscv_compiler_path, "String", "", 
                     "Returns the path to the detected RISC-V compiler");
    
    halt();
}

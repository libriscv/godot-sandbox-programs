#include "mlir_compiler.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <random>
#include <cstdlib>

// MLIR includes
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVM/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Dialect/LLVM/LLVMDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/Vector.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Affine/IR/Affine.h"

// StableHLO includes
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

// LLVM includes
#include "llvm/IR/Module.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>

using namespace mlir;

bool MLIRCompiler::is_mlir_available() {
    try {
        MLIRContext context;
        return true;
    } catch (...) {
        return false;
    }
}

std::string MLIRCompiler::get_mlir_version() {
    // Try to get version from LLVM
    return "MLIR (via LLVM)";
}

bool MLIRCompiler::lower_to_llvm(const std::string &stablehlo_mlir, std::string &llvm_ir) {
    try {
        // Create MLIR context
        MLIRContext context;
        
        // Register all required dialects
        DialectRegistry registry;
        registry.insert<func::FuncDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<math::MathDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<vector::VectorDialect>();
        registry.insert<tensor::TensorDialect>();
        registry.insert<scf::SCFDialect>();
        registry.insert<cf::ControlFlowDialect>();
        registry.insert<LLVM::LLVMDialect>();
        registry.insert<linalg::LinalgDialect>();
        registry.insert<affine::AffineDialect>();
        registry.insert<stablehlo::StablehloDialect>();
        
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();
        
        // Parse StableHLO MLIR
        auto module = parseSourceString<ModuleOp>(stablehlo_mlir, &context);
        if (!module) {
            std::cerr << "Error: Failed to parse StableHLO MLIR" << std::endl;
            return false;
        }
        
        // Create pass manager
        PassManager pm(&context);
        
        // Add StableHLO lowering passes
        // StableHLO -> HLO -> Linalg -> Affine -> Standard -> LLVM
        pm.addNestedPass<func::FuncOp>(stablehlo::createStablehloRefineShapesPass());
        pm.addPass(createStablehloLegalizeToHloPass());
        pm.addPass(createHloLegalizeToLinalgPass());
        pm.addNestedPass<func::FuncOp>(createLinalgLowerToAffineLoopsPass());
        pm.addPass(createLowerAffineToStandardPass());
        pm.addPass(createConvertSCFToCFPass());
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createConvertFuncToLLVMPass());
        pm.addPass(createConvertArithToLLVMPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertMemRefToLLVMPass());
        pm.addPass(createConvertVectorToLLVMPass());
        pm.addPass(createReconcileUnrealizedCastsPass());
        
        // Run passes
        if (failed(pm.run(*module))) {
            std::cerr << "Error: MLIR pass pipeline failed" << std::endl;
            return false;
        }
        
        // Translate to LLVM IR
        llvm::LLVMContext llvmContext;
        registerLLVMDialectTranslation(context);
        
        auto llvmModule = translateModuleToLLVMIR(*module, llvmContext);
        if (!llvmModule) {
            std::cerr << "Error: Failed to translate MLIR to LLVM IR" << std::endl;
            return false;
        }
        
        // Convert LLVM Module to string
        std::string llvm_ir_str;
        llvm::raw_string_ostream os(llvm_ir_str);
        llvmModule->print(os, nullptr);
        llvm_ir = os.str();
        
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error in MLIR lowering: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to create minimal ELF executable from object file
static std::vector<uint8_t> create_minimal_elf_executable(const std::string &obj_path) {
    std::vector<uint8_t> result;
    
    // Read object file
    auto obj_buffer = llvm::MemoryBuffer::getFile(obj_path);
    if (!obj_buffer || !*obj_buffer) {
        std::cerr << "Error: Cannot read object file" << std::endl;
        return result;
    }
    
    auto obj_file = llvm::object::ObjectFile::createObjectFile((*obj_buffer)->getMemBufferRef());
    if (!obj_file || !*obj_file) {
        std::cerr << "Error: Cannot parse object file" << std::endl;
        return result;
    }
    
    // Check if it's an ELF file and get the appropriate type
    llvm::object::Binary *bin = obj_file->get().get();
    if (!bin->isELF()) {
        std::cerr << "Error: Object file is not ELF format" << std::endl;
        return result;
    }
    
    // Try to cast to ELF64 (RISC-V 64-bit)
    using ELF64LE = llvm::object::ELFObjectFile<llvm::object::ELF64LEFile>;
    auto elf_obj = llvm::dyn_cast<ELF64LE>(bin);
    if (!elf_obj) {
        std::cerr << "Error: Object file is not ELF64 format" << std::endl;
        return result;
    }
    
    // Extract sections and symbols from object file
    std::vector<uint8_t> text_data;
    uint64_t entry_point = 0x10000; // Default entry point
    uint64_t text_vaddr = 0x10000;
    
    for (auto section : elf_obj->sections()) {
        if (!section) continue;
        
        auto name = section->getName();
        if (!name) continue;
        
        if (name->str() == ".text") {
            auto contents = section->getContents();
            if (contents) {
                text_data.assign(contents->begin(), contents->end());
            }
            
            // Try to find entry point symbol
            for (auto sym : elf_obj->symbols()) {
                if (!sym) continue;
                auto sym_name = sym->getName();
                if (sym_name && (sym_name->str() == "_start" || sym_name->str() == "main" || sym_name->str() == "_main")) {
                    auto addr = sym->getAddress();
                    if (addr) {
                        entry_point = text_vaddr + (*addr & 0xFFFF);
                    }
                    break;
                }
            }
        }
    }
    
    if (text_data.empty()) {
        std::cerr << "Error: No .text section found in object file" << std::endl;
        return result;
    }
    
    // Create minimal ELF64 executable
    // ELF header (64 bytes)
    result.resize(0x1000); // Start with 4KB
    std::memset(result.data(), 0, result.size());
    
    // ELF header
    result[0] = 0x7f; result[1] = 'E'; result[2] = 'L'; result[3] = 'F'; // ELF magic
    result[4] = 2; // 64-bit
    result[5] = 1; // Little endian
    result[6] = 1; // ELF version
    result[7] = 0; // System V ABI
    result[8] = 0; // ABI version
    // Padding [9-15]
    result[16] = 2; result[17] = 0; // ET_EXEC
    result[18] = 0xF3; result[19] = 0; // RISC-V machine type
    result[20] = 1; result[21] = 0; result[22] = 0; result[23] = 0; // ELF version
    // Entry point (8 bytes, little endian)
    uint64_t entry = entry_point;
    std::memcpy(&result[24], &entry, 8);
    // Program header offset
    uint64_t phoff = 64;
    std::memcpy(&result[32], &phoff, 8);
    // Section header offset (0 for now)
    uint64_t shoff = 0;
    std::memcpy(&result[40], &shoff, 8);
    // Flags
    uint32_t flags = 0;
    std::memcpy(&result[48], &flags, 4);
    // ELF header size
    result[52] = 64; result[53] = 0;
    // Program header size
    result[54] = 56; result[55] = 0;
    // Number of program headers
    result[56] = 1; result[57] = 0;
    // Section header size (0 for now)
    result[58] = 0; result[59] = 0;
    // Number of section headers (0 for now)
    result[60] = 0; result[61] = 0;
    // Section header string table index
    result[62] = 0; result[63] = 0;
    
    // Program header (56 bytes)
    // PT_LOAD
    uint32_t p_type = 1; // PT_LOAD
    std::memcpy(&result[64], &p_type, 4);
    uint32_t p_flags = 5; // PF_R | PF_X (readable and executable)
    std::memcpy(&result[68], &p_flags, 4);
    uint64_t p_offset = 0x1000; // Offset in file
    std::memcpy(&result[72], &p_offset, 8);
    uint64_t p_vaddr = text_vaddr; // Virtual address
    std::memcpy(&result[80], &p_vaddr, 8);
    uint64_t p_paddr = text_vaddr; // Physical address
    std::memcpy(&result[88], &p_paddr, 8);
    uint64_t p_filesz = text_data.size(); // File size
    std::memcpy(&result[96], &p_filesz, 8);
    uint64_t p_memsz = text_data.size(); // Memory size
    std::memcpy(&result[104], &p_memsz, 8);
    uint64_t p_align = 0x1000; // Alignment
    std::memcpy(&result[112], &p_align, 8);
    
    // Copy text section data
    if (result.size() < 0x1000 + text_data.size()) {
        result.resize(0x1000 + text_data.size());
    }
    std::memcpy(&result[0x1000], text_data.data(), text_data.size());
    
    return result;
}

std::vector<uint8_t> MLIRCompiler::compile_llvm_to_riscv_elf64(const std::string &llvm_ir) {
    std::vector<uint8_t> result;
    
    try {
        // Initialize LLVM targets
        llvm::InitializeAllTargetInfos();
        llvm::InitializeAllTargets();
        llvm::InitializeAllTargetMCs();
        llvm::InitializeAllAsmParsers();
        llvm::InitializeAllAsmPrinters();
        
        // Parse LLVM IR
        llvm::LLVMContext context;
        llvm::SMDiagnostic err;
        auto module = llvm::parseIR(llvm::MemoryBufferRef(llvm_ir, "input"), err, context);
        if (!module) {
            std::cerr << "Error: Failed to parse LLVM IR" << std::endl;
            return result;
        }
        
        // Get RISC-V target
        std::string targetTriple = "riscv64-unknown-linux-gnu";
        module->setTargetTriple(targetTriple);
        
        std::string error;
        const llvm::Target *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
        if (!target) {
            std::cerr << "Error: " << error << std::endl;
            return result;
        }
        
        // Create target machine
        llvm::TargetOptions opt;
        // Use a simpler CPU and features that LLVM/clang supports
        // "generic-rv64" might not be recognized, use empty string for generic
        llvm::TargetMachine *targetMachine = target->createTargetMachine(
            targetTriple,
            "",  // Empty CPU string = generic
            "+m,+a,+f,+d",  // Basic RISC-V extensions
            opt,
            llvm::Reloc::Model::Static,
            llvm::CodeModel::Small,
            llvm::CodeGenOptLevel::Aggressive
        );
        
        module->setDataLayout(targetMachine->createDataLayout());
        
        // Generate object file
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 15);
        std::string temp_prefix = "/tmp/mlir_temp_";
        for (int i = 0; i < 8; i++) {
            temp_prefix += "0123456789abcdef"[dis(gen)];
        }
        std::string temp_obj = temp_prefix + ".o";
        
        std::error_code ec;
        llvm::raw_fd_ostream dest(temp_obj, ec, llvm::sys::fs::OF_None);
        if (ec) {
            std::cerr << "Error: Could not open file: " << ec.message() << std::endl;
            return result;
        }
        
        llvm::legacy::PassManager pass;
        auto fileType = llvm::CGFT_ObjectFile;
        if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, fileType)) {
            std::cerr << "Error: TargetMachine can't emit a file of this type" << std::endl;
            dest.close();
            std::filesystem::remove(temp_obj);
            return result;
        }
        
        pass.run(*module);
        dest.flush();
        dest.close();
        
        // Generate minimal ELF executable directly from object file
        result = create_minimal_elf_executable(temp_obj);
        
        // Clean up temporary object file
        std::filesystem::remove(temp_obj);
        
        if (result.empty()) {
            std::cerr << "Error: Failed to create ELF executable" << std::endl;
        }
        
    } catch (const std::exception &e) {
        std::cerr << "Error in LLVM compilation: " << e.what() << std::endl;
    }
    
    return result;
}

std::vector<uint8_t> MLIRCompiler::compile_stablehlo_to_riscv_elf64(const std::string &stablehlo_mlir) {
    std::vector<uint8_t> result;
    
    // Lower StableHLO to LLVM IR
    std::string llvm_ir;
    if (!lower_to_llvm(stablehlo_mlir, llvm_ir)) {
        return result;
    }
    
    // Compile LLVM IR to RISC-V ELF64
    result = compile_llvm_to_riscv_elf64(llvm_ir);
    
    return result;
}

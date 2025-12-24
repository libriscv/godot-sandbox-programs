#include <api.hpp>

#include <compiler.h>
#include <string>
using namespace gdscript;

PUBLIC Variant compile(String code)
{
	// Compile with all debug output
	CompilerOptions options;
	options.dump_tokens = false;
	options.dump_ast = false;
	options.dump_ir = false;
	options.output_elf = true;

	Compiler compiler;
	auto elf_data = compiler.compile(code.utf8(), options);

	if (elf_data.empty()) {
		print("Compilation failed: ", compiler.get_error());
		return PackedByteArray(std::vector<uint8_t>{}); // Return empty array on failure
	}

	PackedByteArray elf(elf_data);
	return elf;
}

int main() {
	// Register the compile function in the API
	ADD_API_FUNCTION(compile, "PackedByteArray", "String code",
		"Compile GDScript to Godot Sandbox program");

	halt();
}

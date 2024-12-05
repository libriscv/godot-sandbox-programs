#include <api.hpp>
#include <cstring>
#include <sys/mman.h>
using AsmCallback = Variant(*)();
struct AsmjitResult {
	unsigned char *data;
	size_t size;
};
extern "C" struct AsmjitResult assemble_to(const char *input, size_t size);

static Variant assemble(String input) {
	// Assemble the input
	const std::string input_str = input.utf8();
	AsmjitResult result = assemble_to(input_str.data(), input_str.size());

	// After simulation, A0 contains assembled program, A1 size
	const uint8_t *program_data = result.data;
	const size_t   size         = result.size;

	void *executable = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	std::memcpy(executable, program_data, size);

	// Create a callable function from the assembled program
	AsmCallback callback = (AsmCallback)executable;

	return Callable::Create<Variant()>(callback);
}

int main() {
	// Add public API
	ADD_API_FUNCTION(assemble, "Callable", "String assembly_code", "Assemble RISC-V assembly code and return a callable function");

	halt();
}

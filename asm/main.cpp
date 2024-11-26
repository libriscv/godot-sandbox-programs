#include <api.hpp>
#include <libriscv/machine.hpp>
#include <sys/mman.h>
using AsmCallback = Variant(*)();
using namespace riscv;
using machine_t = riscv::Machine<riscv::RISCV64>;

EMBED_BINARY(asmjit, "../asmjit/.build/asmjit");
static const std::string_view asmjit_bin(asmjit, asmjit_size);

static Variant assemble(String input) {
	static machine_t master_machine(asmjit_bin, {
		.memory_max = 4UL << 20,
		.stack_size = 256UL << 10,
		.use_memory_arena = false,
	});
	machine_t machine(master_machine, {
		.memory_max = 4UL << 20,
		.stack_size = 256UL << 10,
		.use_memory_arena = false,
	});
	machine.setup_linux_syscalls(false, false);
	machine.setup_linux({"asmjit", input.utf8(), "stuff"}, {"LC_ALL=C", "USER=guest"});

	machine.cpu.simulate_inaccurate(machine.cpu.pc());

	const uint64_t data = machine.cpu.reg(10);
	const size_t   size = machine.cpu.reg(11);
	if (data == 0) {
		fprintf(stderr, "Failed to assemble program\n");
		return Variant();
	}

	// After simulation, A0 contains assembled program, A1 size
	std::string_view program = machine.memory.memview(data, size);
	const char *program_data = program.data();

	void *executable = mmap(nullptr, size, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	std::memcpy(executable, program_data, size);

	// Create a callable function from the assembled program
	AsmCallback callback = (AsmCallback)executable;

	return Callable::Create<Variant()>(callback);
}

SANDBOX_API({
	.name = "assemble",
	.address = (void*)&assemble,
	.description = "Assemble RISC-V assembly code and return a callable function",
	.return_type = "Callable",
	.arguments = "String assembly_code",
});

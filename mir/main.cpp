#include <api.hpp>
#include <cstring>
extern "C" {
#include <mir-gen.h>
#include <c2mir/c2mir.h>
}
#include <cstdarg>
EXTERN_SYSCALL(void, sys_vstore, unsigned, const void *, size_t);
EXTERN_SYSCALL(void, sys_vfetch, unsigned, void *, int);

struct Data {
	const char *code;
	const char *code_end;
};

static int get_cfunc(void *opaque) {
	Data *data_ptr = (Data *)opaque;

	if (data_ptr->code >= data_ptr->code_end)
		return EOF;
	return *data_ptr->code++;
}
static MIR_context_t ctx;
static const int optlevel = 2;

static void *import_resolver(const char *name) {
	if (strcmp(name, "sys_print") == 0) {
		return (void *)sys_print;
	} else if (strcmp(name, "sys_vfetch") == 0) {
		return (void *)sys_vfetch;
	} else if (strcmp(name, "sys_vstore") == 0) {
		return (void *)sys_vstore;
	} else if (strcmp(name, "malloc") == 0) {
		return (void *)malloc;
	} else if (strcmp(name, "free") == 0) {
		return (void *)free;
	}
	printf("import_resolver missing: %s\n", name);
	fflush(stdout);
	return nullptr;
}

static MIR_item_t mir_find_function(MIR_module_t module, const char *func_name) {
	MIR_item_t func, func_item = NULL;
	for (func = DLIST_HEAD(MIR_item_t, module->items); func != NULL; func = DLIST_NEXT(MIR_item_t, func)) {
		if (func->item_type == MIR_func_item && strcmp(func->u.func->name, func_name) == 0) {
			func_item = func;
			break;
		}
	}
	return func_item;
}

static void *mir_get_func(MIR_context_t ctx, MIR_module_t module, const char *func_name) {
	MIR_item_t func_item = mir_find_function(module, func_name);
	if (func_item == NULL) {
		fprintf(stderr, "Error: Mir function %s not found\n", func_name);
		fflush(stdout);
		return NULL;
	}
	return MIR_gen(ctx, func_item);
}

static __attribute__((noreturn)) void error_func(MIR_error_type_t error_type, const char *format, ...) {
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	fflush(stdout);
	abort();
}

int main() {
	ctx = MIR_init();
	MIR_set_error_func(ctx, error_func);

	halt();
}

#define VERBOSE_COMPILE 0

static Variant do_compile(const std::string &source_code, const std::string &entry) {
	std::array<const char*, 0> include_dirs = {
	};
	std::array<c2mir_macro_command, 1> macro_commands = {
		{ 1, "TEST", "1" }
	};

#if VERBOSE_COMPILE
	printf("Compiling C code: %s\n", source_code.c_str());
	fflush(stdout);
#endif

	// Add our own API
	std::string code = R"(
	extern void sys_vfetch(unsigned, void *, int);
	extern void sys_vstore(unsigned *, int, const void *, unsigned);
	extern void print_int(int);
	extern void print_float(float);
	extern void print_string(const char*);
	extern void print_ptr(void*);
	extern void *malloc(unsigned long);
	extern void free(void*);
	struct Variant {
		long type;
		long value;
	};
	struct VectorF32 {
		float *f_begin;
		float *f_end;
		float *f_cap;
	};
)";
	code += source_code;

	Data data;
	data.code = code.c_str();
	data.code_end = code.c_str() + code.size();

	c2mir_init(ctx);
	c2mir_options ops;
	memset(&ops, 0, sizeof(ops));
	ops.message_file = stdout;
	ops.output_file_name = nullptr;
	ops.include_dirs_num = include_dirs.size();
	ops.include_dirs = include_dirs.data();
	ops.macro_commands_num = macro_commands.size();
	ops.macro_commands = macro_commands.data();
	int result = c2mir_compile(ctx, &ops, &get_cfunc, (void*)&data, "test.c", nullptr);
	if (!result) {
		printf("Failed to compile C code\n");
		fflush(stdout);
		return Nil;
	}

#if VERBOSE_COMPILE
	printf("*** Compilation successful\n");
	fflush(stdout);
#endif
	c2mir_finish(ctx);

	auto *module = DLIST_TAIL(MIR_module_t, *MIR_get_module_list(ctx));
	if (!module) {
		fprintf(stderr, "No module found\n");
		fflush(stdout);
		return Nil;
	}
	MIR_gen_init(ctx);
	MIR_gen_set_optimize_level(ctx, optlevel);
	MIR_load_module(ctx, module);
	MIR_load_external(ctx, "memset", (void *)memset);
	MIR_load_external(ctx, "memcpy", (void *)memcpy);
	MIR_link(ctx, MIR_set_gen_interface, import_resolver);

	Variant (*fun_addr)() = NULL;
	fun_addr = (Variant(*)())mir_get_func(ctx, module, entry.c_str());

	// Finish the code-generation
	//MIR_gen_finish(ctx);

	if (!fun_addr) {
		fprintf(stderr, "Function %s not found\n", entry.c_str());
		fflush(stdout);
		return Nil;
	}
#if VERBOSE_COMPILE
	printf("Function %s found, address %p\n", entry.c_str(), fun_addr);
	fflush(stdout);
#endif

	// Return a callable function
	return Callable::Create<Variant()>(fun_addr);
}

extern "C" Variant compile(String code, String entry) {
	const std::string utf = code.utf8();
	const std::string entry_utf = entry.utf8();

	return do_compile(utf, entry);
}

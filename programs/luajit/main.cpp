#include <api.hpp>
#include <cstring>
extern "C" {
#include <luajit-2.1/lauxlib.h>
#include <luajit-2.1/lua.h>
#include <luajit-2.1/lualib.h>
}

//#define CREATE_MENU_BOX 1
#ifdef CREATE_MENU_BOX
static int api_print(lua_State *L) {
	const char *text = luaL_checkstring(L, 1);
	get_node("../TextAnswer")("set_text", text);
	return 0;
}
extern "C" Variant click() {
	String text = get_node("../TextEdit")("get_text");
	return run(text);
}
#else
static int api_print(lua_State *L) {
	const char *text = luaL_checkstring(L, 1);
	printf("%s", text);
	fflush(stdout);
	return 0;
}
#endif

static lua_State *L;
static constexpr bool VERBOSE = false;

static Variant run(String code) {
	// Load a string as a script
	const std::string utf = code.utf8();
	luaL_loadbuffer(L, utf.c_str(), utf.size(), "@code");

	// Run the script (0 arguments, 1 result)
	lua_pcall(L, 0, 1, 0);

	// Get the result type
	const int type = lua_type(L, -1);
	switch (type) {
		case LUA_TNIL:
			return Nil;
		case LUA_TBOOLEAN:
			return lua_toboolean(L, -1);
		case LUA_TNUMBER:
			return lua_tonumber(L, -1);
		case LUA_TSTRING:
			return lua_tostring(L, -1);
		default:
			return Nil;
	}
}

static Variant add_function(String function_name, Callable function) {
	// Create a struct to store callback information
	struct UserData {
		Variant function;
	};
	UserData *data = new UserData;
	// Make sure the function is not garbage collected, store it in the struct
	data->function = function;
	data->function.make_permanent();
	// Push the function to the Lua stack
	lua_pushlightuserdata(L, (void *)data);
	lua_pushcclosure(L, [](lua_State *L) -> int {
		UserData *data = (UserData *)lua_touserdata(L, lua_upvalueindex(1));
		if constexpr (VERBOSE) {
			printf("Calling function with Callable Variant type %d index %d\n",
				data->function.get_type(), data->function.get_internal_index());
		}
		Variant &function = data->function;
		// Create a fixed-size array to store the arguments
		std::array<Variant, 8> args;
		size_t arg_count = 0;
		// Find the number of arguments
		const int nargs = lua_gettop(L);
		for (int i = 1; i <= nargs; i++) {
			// Push the arguments to the vector
			switch (lua_type(L, i)) {
				case LUA_TNIL:
					break;
				case LUA_TBOOLEAN:
					if constexpr (VERBOSE)
						printf("Boolean argument %d\n", lua_toboolean(L, i));
					args.at(arg_count++) = bool(lua_toboolean(L, i));
					break;
				case LUA_TNUMBER:
					if constexpr (VERBOSE)
						printf("Number argument %f\n", lua_tonumber(L, i));
					args.at(arg_count++) = double(lua_tonumber(L, i));
					break;
				case LUA_TSTRING:
					if constexpr (VERBOSE)
						printf("String argument %s\n", lua_tostring(L, i));
					args.at(arg_count++) = lua_tostring(L, i);
					break;
				default:
					if constexpr (VERBOSE)
						printf("Unknown argument type %d\n", lua_type(L, i));
					break;
			}
		}
		if constexpr (VERBOSE) {
			printf("Calling function with %zu arguments\n", arg_count);
			for (size_t i = 0; i < arg_count; i++) {
				printf("Argument %zu type %d\n", i, args.at(i).get_type());
			}
			fflush(stdout);
		}
		Variant result;
		function.callp("call", args.data(), arg_count, result);
		// Push the result to the Lua stack
		switch (result.get_type()) {
			case Variant::Type::NIL:
				return 0;
			case Variant::Type::BOOL:
				lua_pushboolean(L, result);
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				lua_pushnumber(L, result);
				return 1;
			case Variant::Type::STRING:
				lua_pushstring(L, result.as_std_string().c_str());
				return 1;
			default:
				return 0;
		}
	}, 1);
	lua_setglobal(L, function_name.utf8().c_str());
	return Nil;
}

int main() {
#ifdef CREATE_MENU_BOX
	// Activate this mod
	get_parent().call("set_visible", true);
	get_node("../Button").connect("pressed", Callable(click));

	CallbackTimer::native_periodic(0.0125, [](Node timer) -> Variant {
		Node2D mod = get_parent(); // From the Timers POV
		static Vector2 origin = mod.get_position();
		static constexpr float period = 2.0f;
		static float x = 0.0f;
		const float progress = 1.0f - x / 4.0f;
		if (progress <= 0.0f) {
			timer.queue_free();
		}

		const float anim = (Math::sin(x * period + x) * 2.0f - 1.0f) * 0.1f * progress;
		const Vector2 scale(1.0f + anim);
		mod.set_position(origin - scale * 55.0f);
		mod.set_scale(scale);
		x += 0.1f;
		return Nil;
	});
#endif

	L = luaL_newstate();

	luaL_openlibs(L); /* Load Lua libraries */

	// API bindings
	lua_register(L, "print", api_print);

	ADD_API_FUNCTION(run, "Variant", "String code");
	ADD_API_FUNCTION(add_function, "void", "String function_name, Callable function");

	halt();
}

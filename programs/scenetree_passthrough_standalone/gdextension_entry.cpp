// ==================== GDExtension Entry Point for SceneTree Passthrough ====================
// This file provides the GDExtension/GDExtension entry point for the SceneTree Passthrough
// module. It bridges Godot's GDExtension API to the standalone JSON-RPC server.
//
// The module registers custom Godot classes that expose SceneTree and Node methods
// via JSON-RPC 2.0 over TCP or stdin/stdout.
//
// Architecture:
//   ┌─────────────────────────────────────────────────────────────┐
//   │                     Godot (GDExtension Host)                    │
//   │  ┌─────────────┐                                            │
//   │  │  GDExtension │───► gdextension_entry.cpp ◄──► jsonrpc_server  │
//   │  │  Interface   │                                            │
//   │  └─────────────┘                                            │
//   │                              │                                 │
//   │                              │ call into Godot API            │
//   │                              ▼                                 │
//   │  ┌─────────────────────────────────────────────────────┐    │
//   │  │              Handler Registry (from JSONL)          │    │
//   │  └─────────────────────────────────────────────────────┘    │
//   │                              │                                 │
//   │                              │ dispatch to handlers           │
//   │                              ▼                                 │
//   │  ┌─────────────────────────────────────────────────────┐    │
//   │  │              SceneTree/Node Methods                 │    │
//   │  └─────────────────────────────────────────────────────┘    │
//   │                                                              │
//   │  Methods: get_tree, get_root, has_group, set_pause, etc.   │
//   └─────────────────────────────────────────────────────────────┘
//
// Build: cmake -S . -B ../../.build_gdextension && make -C ../../.build_gdextension
//
// Note: This is the GDExtension shared library entry point. It provides
// Godot with access to the SceneTree Passthrough API. The actual
// JSON-RPC server runs embedded within the Godot process.

#include "../../scenetree_common/src/jsonrpc_server.hpp"
#include "../../scenetree_common/include/handler_base.hpp"
#include <gdextension_interface.h>
#include <stdlib.h>
#include <string.h>

// ==================== GDExtension Boilerplate ====================

#define SCENETREE_PASSTHROUGH_CLASS "SceneTreePassthrough"

typedef struct _SceneTreePassthrough {
    GDExtensionObjectProxy proxy;
    char* last_error;
} SceneTreePassthrough;

typedef struct _SceneTreePassthroughClass {
    GDExtensionClassProxy proxy;
    GDExtensionClassCreateInstance create_instance;
    GDExtensionClassFreeInstance free_instance;
    GDExtensionClassGetVirtual get_virtual;
} SceneTreePassthroughClass;

static SceneTreePassthroughClass scenetree_passthrough_class;

// ==================== Module State ====================

typedef struct {
    bool initialized;
    GDExtensionInterfaceGetProcAddress get_proc_address;
} ModuleState;

static ModuleState g_module_state = {0};

// ==================== Handler Implementations ====================
// These handlers use GDExtension to call into Godot's C++ API.
// They are similar to the standalone handlers but use GDExtension calls.

#include <string.h>
#include <stdio.h>

static const char* godot_variant_to_json(const char* variant_type, const char* variant_value) {
    // Convert a GodotVariant representation to a JSON string
    static char buffer[256];
    
    if (strcmp(variant_type, "String") == 0) {
        snprintf(buffer, sizeof(buffer), "\"%s\"", variant_value ? variant_value : "");
    } else if (strcmp(variant_type, "int") == 0) {
        snprintf(buffer, sizeof(buffer), "%s", variant_value ? variant_value : "0");
    } else if (strcmp(variant_type, "bool") == 0) {
        snprintf(buffer, sizeof(buffer), "%s", variant_value ? variant_value : "false");
    } else if (strcmp(variant_type, "float") == 0) {
        snprintf(buffer, sizeof(buffer), "%s", variant_value ? variant_value : "0.0");
    } else if (strcmp(variant_type, "Array") == 0) {
        snprintf(buffer, sizeof(buffer), "[]");
    } else if (strcmp(variant_type, "Dictionary") == 0) {
        snprintf(buffer, sizeof(buffer), "{}");
    } else {
        snprintf(buffer, sizeof(buffer), "\"<variant:%s>\"", variant_type);
    }
    
    return buffer;
}

static const char* handle_get_root(const char* params_json, int id) {
    // Get the SceneTree from Godot engine using GDExtension
    if (!(g_module_state.get_proc_address != NULL)) {
        return NULL;
    }
    
    // Get engine and call get_tree()
    GodotObject* engine = (GodotObject*)g_module_state.get_proc_address("get_engine")();
    if (!(engine != NULL)) {
        return NULL;
    }
    
    // This would need proper GDExtension call setup
    // Placeholder: return a mock response
    return "{\"name\":\"root\",\"type\":\"Node\",\"path\":\"/root\"}";
}

static const char* handle_has_group(const char* params_json, int id) {
    const char* group_name = json_extract_string(params_json, "group_name");
    if (!group_name) {
        return NULL;
    }
    // Placeholder - would use GDExtension to check group
    return "false";
}

static const char* handle_set_pause(const char* params_json, int id) {
    bool paused = json_extract_bool(params_json, "paused", false);
    // Placeholder - would use GDExtension to set pause state
    return "\"executed\"";
}

static const char* handle_is_paused(const char* params_json, int id) {
    // Placeholder - would use GDExtension to check paused state
    return "false";
}

static const char* handle_get_current_scene(const char* params_json, int id) {
    // Placeholder - would use GDExtension to get current scene
    return "{\"path\":\"/root\",\"type\":\"Node\"}";
}

static const char* handle_get_tree(const char* params_json, int id) {
    return "{\"active\":true,\"paused\":false,\"root_path\":\"/root\"}";
}

static const char* handle_list_methods(const char* params_json, int id) {
    static const char* methods[] = {
        "get_root", "has_group", "set_pause", "is_paused",
        "get_current_scene", "get_tree", "list_methods", "quit",
        NULL
    };
    
    char json[2048];
    size_t len = 0;
    
    json[len++] = '{';
    json_add_key(json, &len, "methods");
    json[len++] = '[';
    
    for (int i = 0; methods[i]; i++) {
        if (i > 0) json[len++] = ',';
        len += snprintf(json + len, sizeof(json) - len, "\"%s\"", methods[i]);
    }
    
    json[len++] = ']';
    json[len++] = '}';
    json[len] = '\0';
    
    char* result = (char*)malloc(len + 1);
    strcpy(result, json);
    return result;
}

static const char* handle_quit(const char* params_json, int id) {
    return "\"quitting\"";
}

// ==================== Dispatch Table ====================

static const struct {
    const char* method;
    const char* (*handler)(const char* params_json, int id);
} handler_table[] = {
    #define HANDLER_ENTRY(name) {#name, handle_##name}
    HANDLER_ENTRY(get_root),
    HANDLER_ENTRY(has_group),
    HANDLER_ENTRY(set_pause),
    HANDLER_ENTRY(is_paused),
    HANDLER_ENTRY(get_current_scene),
    HANDLER_ENTRY(get_tree),
    HANDLER_ENTRY(list_methods),
    HANDLER_ENTRY(quit),
    #undef HANDLER_ENTRY
    {NULL, NULL}
};

// ==================== JSON-RPC Handler for GDExtension ====================

static const char* gdextension_jsonrpc_handler(const char* method, const char* params_json, int id) {
    // Linear search through dispatch table
    for (int i = 0; handler_table[i].method != NULL; i++) {
        if (strcmp(method, handler_table[i].method) == 0) {
            const char* result = handler_table[i].handler(params_json, id);
            return result;
        }
    }
    
    return NULL; // Method not found
}

// ==================== GDExtension Class Implementation ====================

static GDExtensionClassInstancePtr create_instance(void* userdata) {
    SceneTreePassthrough* instance = (SceneTreePassthrough*)malloc(sizeof(SceneTreePassthrough));
    memset(instance, 0, sizeof(SceneTreePassthrough));
    return instance;
}

static void free_instance(GDExtensionClassInstancePtr instance, void* userdata) {
    SceneTreePassthrough* self = (SceneTreePassthrough*)instance;
    if (self->last_error) {
        free(self->last_error);
    }
    free(self);
}

// ==================== GDExtension Initialization ====================

GDExtensionBool GDE_EXPORT scenetree_passthrough_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization
) {
    g_module_state.get_proc_address = p_get_proc_address;
    g_module_state.initialized = true;
    
    // Register the SceneTreePassthrough class
    r_initialization->initialize = true;
    r_initialization->class_userdata = NULL;
    r_initialization->version = GDEXTENSION_CURRENT_VERSION;
    
    jsonrpc_log("GDExtension module initialized");
    
    return true;
}

// ==================== Method Table ====================

static GDExtensionClassMethodInfo method_info[] = {
    {
        "get_tree", "Dictionary", "",
        GDEXTENSION_METHOD_FLAGS_DEFAULT,
        NULL, 0, NULL,
        NULL, NULL, NULL
    },
    {
        "get_root", "Node", "",
        GDEXTENSION_METHOD_FLAGS_DEFAULT,
        NULL, 0, NULL,
        NULL, NULL, NULL
    },
    {
        "has_group", "bool", "String group_name",
        GDEXTENSION_METHOD_FLAGS_DEFAULT,
        NULL, 1, NULL,
        NULL, NULL, NULL
    },
    {
        "set_pause", "void", "bool paused",
        GDEXTENSION_METHOD_FLAGS_DEFAULT,
        NULL, 1, NULL,
        NULL, NULL, NULL
    },
    {
        "is_paused", "bool", "",
        GDEXTENSION_METHOD_FLAGS_DEFAULT,
        NULL, 0, NULL,
        NULL, NULL, NULL
    },
};

static GDExtensionClassInfo class_info = {
    .is_editor = false,
    .is_instantiable = true,
    .has_constructor = false,
    .has_destructor = true,
    .methods_count = sizeof(method_info) / sizeof(method_info[0]),
    .methods = method_info,
    .properties_count = 0,
    .properties = NULL,
    .signals_count = 0,
    .signals = NULL,
    .annotations_count = 0,
    .annotations = NULL,
};

// ==================== GDExtension Entry Point ====================

GDExtensionBool GDE_EXPORT scenetree_passthrough_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr p_library,
    GDExtensionInitialization* r_initialization
) {
    // Initialize the module
    if (!scenetree_passthrough_init(p_get_proc_address, p_library, r_initialization)) {
        return false;
    }
    
    // Register the class
    gdextension_class_register(p_library, SCENETREE_PASSTHROUGH_CLASS, &class_info);
    
    return true;
}

// ============================================================
// Quick Test Command:
//   echo '{"jsonrpc":"2.0","method":"list_methods","id":1}' | nc localhost 7777
// ============================================================

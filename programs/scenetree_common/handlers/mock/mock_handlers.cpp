// ==================== Mock SceneTree Handlers ====================
// Mock implementations for standalone/stdio testing.
// These return plausible responses without requiring Godot API access.

#include "../../include/handler_base.hpp"
#include <stdlib.h>  // for asprintf

// ==================== Mock State ====================

static int mock_next_node_id = 1;

// ==================== Error Helpers ====================

static const char* make_error(int code, const char* message) {
    char* error_response;
    asprintf(&error_response,
        "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":%d,\"message\":\"%s\"},\"id\":null}",
        code, message);
    return error_response;
}

static const char* make_result(const char* json_value) {
    char* result_response;
    asprintf(&result_response,
        "{\"jsonrpc\":\"2.0\",\"result\":%s,\"id\":null}",
        json_value ? json_value : "null");
    return result_response;
}

static const char* make_ok_result() {
    return make_result("true");
}

// ==================== Method Handlers ====================

// Mock: get_root - returns root node info
const char* handle_mock_get_root(const char* params_json, int id) {
    char* response;
    asprintf(&response,
        "{\"name\":\"root\",\"type\":\"Node\",\"path\":\"/root\",\"id\":0}");
    return make_result(response);
}

// Mock: get_node - returns node info for given path
const char* handle_mock_get_node(const char* params_json, int id) {
    const char* path = json_extract_string(params_json, "path");
    if (!path || !*path) {
        return make_error(JSONRPC_INVALID_PARAMS, "Missing 'path' parameter");
    }
    
    char* response;
    asprintf(&response,
        "{\"path\":\"%s\",\"name\":\"node_%s\",\"type\":\"Node\",\"exists\":true}",
        path, path);
    return make_result(response);
}

// Mock: list_methods - returns available methods
const char* handle_mock_list_methods(const char* params_json, int id) {
    const char* response = 
        "[\[\"get_root\"\],"
        " \"get_node\","
        " \"list_methods\","
        " \"get_tree\","
        " \"reload_current_scene\","
        " \"create_node\","
        " \"delete_node\","
        " \"get_node_count\"]";
    return make_result(response);
}

// Mock: echo - echoes back parameters
const char* handle_mock_echo(const char* params_json, int id) {
    // Return params as-is (useful for testing)
    char* response;
    asprintf(&response, "%s", params_json);
    return make_result(response);
}

// Mock: create_node - creates a mock node
const char* handle_mock_create_node(const char* params_json, int id) {
    const char* path = json_extract_string(params_json, "path");
    const char* type = json_extract_string(params_json, "type");
    
    if (!path || !*path) {
        return make_error(JSONRPC_INVALID_PARAMS, "Missing 'path' parameter");
    }
    
    char* response;
    asprintf(&response,
        "{\"path\":\"%s\",\"type\":\"%s\",\"created\":true,\"node_id\":%d}",
        path, type ? type : "Node", mock_next_node_id++);
    return make_result(response);
}

// Mock: get_tree - returns tree structure
const char* handle_mock_get_tree(const char* params_json, int id) {
    const char* response =
        "{\"root\":{\"name\":\"root\",\"path\":\"/root\","
        "\"children\":[",
        "{\"name\":\"child1\",\"path\":\"/root/child1\"},",
        "{\"name\":\"child2\",\"path\":\"/root/child2\"}",
        "]}}";
    return make_result(response);
}

// Mock: reload_current_scene - reloads scene
const char* handle_mock_reload_scene(const char* params_json, int id) {
    char* response;
    asprintf(&response,
        "{\"reloaded\":true,\"scene_name\":\"current_scene.tscn\"}");
    return make_result(response);
}

// Mock: get_node_count - returns node count
const char* handle_mock_get_node_count(const char* params_json, int id) {
    char* response;
    asprintf(&response, "{\"count\":%d}", mock_next_node_id - 1);
    return make_result(response);
}

// Mock: dummy_handler - placeholder for unimplemented handlers
const char* handle_mock_dummy(const char* params_json, int id) {
    return make_result("\"ok\"");
}

// ==================== Dispatch Hook ====================
// This is the entry point called by the JSON-RPC server

#ifdef __cplusplus
extern "C" {
#endif

// Dispatch table mapping method names to handlers
// This is used by the jsonrpc_server when a request arrives

#include <string.h>

// Define handler type
typedef const char* (*MockHandlerFunc)(const char* params_json, int id);

typedef struct {
    const char* method;
    MockHandlerFunc handler;
} MockMethodMap;

static const MockMethodMap mock_method_map[] = {
    {"get_root", handle_mock_get_root},
    {"get_node", handle_mock_get_node},
    {"list_methods", handle_mock_list_methods},
    {"echo", handle_mock_echo},
    {"create_node", handle_mock_create_node},
    {"get_tree", handle_mock_get_tree},
    {"reload_current_scene", handle_mock_reload_scene},
    {"get_node_count", handle_mock_get_node_count},
    // Catch-all for unimplemented methods
    {NULL, handle_mock_dummy}
};

// Main dispatch function - called by jsonrpc_server when a request arrives
const char* mock_dispatch_handler(const char* method, const char* params_json, int id) {
    if (!method || !*method) {
        return make_error(JSONRPC_INVALID_REQUEST, "Missing method");
    }
    
    // Linear search through dispatch table
    for (int i = 0; mock_method_map[i].method != NULL; i++) {
        if (strcmp(method, mock_method_map[i].method) == 0) {
            return mock_method_map[i].handler(params_json, id);
        }
    }
    
    // Method not found
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "Method '%s' not found", method);
    return make_error(JSONRPC_METHOD_NOT_FOUND, error_msg);
}

#ifdef __cplusplus
}
#endif

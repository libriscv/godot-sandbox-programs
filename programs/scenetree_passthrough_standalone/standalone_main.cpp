#include "handlers.hpp"
#include <climits>
#include <vector>
#include <utility>

// Standalone variant for x86_64 native execution
// This version initializes Godot and starts JSON-RPC server directly,
// without requiring a .tscn scene file or godot-sandbox RISC-V VM.

// Global handler instance
static JSONRPCHandler handler;
static Object jsonrpc_tcp_server(0);
static std::vector<Object> jsonrpc_clients;
static bool jsonrpc_server_started = false;
static int jsonrpc_server_port = 7777;

#define DECLARE_RPC_WRAPPER(name) \
    PUBLIC Dictionary name(const Array& callv_args) { \
        return handler.handle_##name(callv_args); \
    }

DECLARE_RPC_WRAPPER(_initialize)
DECLARE_RPC_WRAPPER(_physics_process)
DECLARE_RPC_WRAPPER(_process)
DECLARE_RPC_WRAPPER(_finalize)
DECLARE_RPC_WRAPPER(_init)
DECLARE_RPC_WRAPPER(_to_string)
DECLARE_RPC_WRAPPER(_notification)
DECLARE_RPC_WRAPPER(_set)
DECLARE_RPC_WRAPPER(_get)
DECLARE_RPC_WRAPPER(_get_property_list)
DECLARE_RPC_WRAPPER(_validate_property)
DECLARE_RPC_WRAPPER(_property_can_revert)
DECLARE_RPC_WRAPPER(_property_get_revert)
DECLARE_RPC_WRAPPER(_iter_init)
DECLARE_RPC_WRAPPER(_iter_next)
DECLARE_RPC_WRAPPER(_iter_get)

#undef DECLARE_RPC_WRAPPER

// JSON-RPC dispatcher implementation - routes requests to appropriate handlers
static Dictionary jsonrpc_dispatch_impl(const Dictionary& request) {
    Variant method_value = request.get("method");
    String method_name = method_value.get_type() == Variant::STRING ? String(method_value) : String();
    
    if (method_name.is_empty()) {
        Dictionary error_response;
        Dictionary error;
        error["code"] = -32600;
        error["message"] = "Invalid request";
        error_response["jsonrpc"] = "2.0";
        error_response["error"] = error;
        return error_response;
    }
    
    // Extract parameters as Array
    Array params;
    if (request.has("params")) {
        Variant params_value = request.get("params");
        if (params_value.get_type() == Variant::ARRAY) {
            params = Array(params_value);
        }
    }
    
    // Route to appropriate handler
    if (method_name == "_initialize") return handler.handle__initialize(params);
    if (method_name == "_physics_process") return handler.handle__physics_process(params);
    if (method_name == "_process") return handler.handle__process(params);
    if (method_name == "_finalize") return handler.handle__finalize(params);
    if (method_name == "_init") return handler.handle__init(params);
    if (method_name == "_to_string") return handler.handle__to_string(params);
    if (method_name == "_notification") return handler.handle__notification(params);
    if (method_name == "_set") return handler.handle__set(params);
    if (method_name == "_get") return handler.handle__get(params);
    if (method_name == "_get_property_list") return handler.handle__get_property_list(params);
    if (method_name == "_validate_property") return handler.handle__validate_property(params);
    if (method_name == "_property_can_revert") return handler.handle__property_can_revert(params);
    if (method_name == "_property_get_revert") return handler.handle__property_get_revert(params);
    if (method_name == "_iter_init") return handler.handle__iter_init(params);
    if (method_name == "_iter_next") return handler.handle__iter_next(params);
    if (method_name == "_iter_get") return handler.handle__iter_get(params);
    if (method_name == "add_user_signal") return handler.handle_add_user_signal(params);
    if (method_name == "call") return handler.handle_call(params);
    if (method_name == "call_deferred") return handler.handle_call_deferred(params);
    if (method_name == "call_group") return handler.handle_call_group(params);
    if (method_name == "call_group_flags") return handler.handle_call_group_flags(params);
    if (method_name == "callv") return handler.handle_callv(params);
    if (method_name == "can_translate_messages") return handler.handle_can_translate_messages(params);
    if (method_name == "cancel_free") return handler.handle_cancel_free(params);
    if (method_name == "change_scene_to_file") return handler.handle_change_scene_to_file(params);
    if (method_name == "change_scene_to_node") return handler.handle_change_scene_to_node(params);
    if (method_name == "change_scene_to_packed") return handler.handle_change_scene_to_packed(params);
    if (method_name == "connect") return handler.handle_connect(params);
    if (method_name == "create_timer") return handler.handle_create_timer(params);
    if (method_name == "create_tween") return handler.handle_create_tween(params);
    if (method_name == "disconnect") return handler.handle_disconnect(params);
    if (method_name == "emit_signal") return handler.handle_emit_signal(params);
    if (method_name == "free") return handler.handle_free(params);
    if (method_name == "get") return handler.handle_get(params);
    if (method_name == "get_class") return handler.handle_get_class(params);
    if (method_name == "get_current_scene") return handler.handle_get_current_scene(params);
    if (method_name == "get_edited_scene_root") return handler.handle_get_edited_scene_root(params);
    if (method_name == "get_first_node_in_group") return handler.handle_get_first_node_in_group(params);
    if (method_name == "get_frame") return handler.handle_get_frame(params);
    if (method_name == "get_incoming_connections") return handler.handle_get_incoming_connections(params);
    if (method_name == "get_indexed") return handler.handle_get_indexed(params);
    if (method_name == "get_instance_id") return handler.handle_get_instance_id(params);
    if (method_name == "get_meta") return handler.handle_get_meta(params);
    if (method_name == "get_meta_list") return handler.handle_get_meta_list(params);
    if (method_name == "get_method_argument_count") return handler.handle_get_method_argument_count(params);
    if (method_name == "get_method_list") return handler.handle_get_method_list(params);
    if (method_name == "get_multiplayer") return handler.handle_get_multiplayer(params);
    if (method_name == "get_node_count") return handler.handle_get_node_count(params);
    if (method_name == "get_node_count_in_group") return handler.handle_get_node_count_in_group(params);
    if (method_name == "get_nodes_in_group") return handler.handle_get_nodes_in_group(params);
    if (method_name == "get_processed_tweens") return handler.handle_get_processed_tweens(params);
    if (method_name == "get_property_list") return handler.handle_get_property_list(params);
    if (method_name == "get_root") return handler.handle_get_root(params);
    if (method_name == "get_script") return handler.handle_get_script(params);
    if (method_name == "get_signal_connection_list") return handler.handle_get_signal_connection_list(params);
    if (method_name == "get_signal_list") return handler.handle_get_signal_list(params);
    if (method_name == "get_translation_domain") return handler.handle_get_translation_domain(params);
    if (method_name == "has_connections") return handler.handle_has_connections(params);
    if (method_name == "has_group") return handler.handle_has_group(params);
    if (method_name == "has_meta") return handler.handle_has_meta(params);
    if (method_name == "has_method") return handler.handle_has_method(params);
    if (method_name == "has_signal") return handler.handle_has_signal(params);
    if (method_name == "has_user_signal") return handler.handle_has_user_signal(params);
    if (method_name == "is_accessibility_enabled") return handler.handle_is_accessibility_enabled(params);
    if (method_name == "is_accessibility_supported") return handler.handle_is_accessibility_supported(params);
    if (method_name == "is_auto_accept_quit") return handler.handle_is_auto_accept_quit(params);
    if (method_name == "is_blocking_signals") return handler.handle_is_blocking_signals(params);
    if (method_name == "is_class") return handler.handle_is_class(params);
    if (method_name == "is_connected") return handler.handle_is_connected(params);
    if (method_name == "is_debugging_collisions_hint") return handler.handle_is_debugging_collisions_hint(params);
    if (method_name == "is_debugging_navigation_hint") return handler.handle_is_debugging_navigation_hint(params);
    if (method_name == "is_debugging_paths_hint") return handler.handle_is_debugging_paths_hint(params);
    if (method_name == "is_multiplayer_poll_enabled") return handler.handle_is_multiplayer_poll_enabled(params);
    if (method_name == "is_paused") return handler.handle_is_paused(params);
    if (method_name == "is_physics_interpolation_enabled") return handler.handle_is_physics_interpolation_enabled(params);
    if (method_name == "is_queued_for_deletion") return handler.handle_is_queued_for_deletion(params);
    if (method_name == "is_quit_on_go_back") return handler.handle_is_quit_on_go_back(params);
    if (method_name == "notification") return handler.handle_notification(params);
    if (method_name == "notify_group") return handler.handle_notify_group(params);
    if (method_name == "notify_group_flags") return handler.handle_notify_group_flags(params);
    if (method_name == "notify_property_list_changed") return handler.handle_notify_property_list_changed(params);
    if (method_name == "property_can_revert") return handler.handle_property_can_revert(params);
    if (method_name == "property_get_revert") return handler.handle_property_get_revert(params);
    if (method_name == "queue_delete") return handler.handle_queue_delete(params);
    if (method_name == "quit") return handler.handle_quit(params);
    if (method_name == "reload_current_scene") return handler.handle_reload_current_scene(params);
    if (method_name == "remove_meta") return handler.handle_remove_meta(params);
    if (method_name == "remove_user_signal") return handler.handle_remove_user_signal(params);
    if (method_name == "set") return handler.handle_set(params);
    if (method_name == "set_auto_accept_quit") return handler.handle_set_auto_accept_quit(params);
    if (method_name == "set_block_signals") return handler.handle_set_block_signals(params);
    if (method_name == "set_current_scene") return handler.handle_set_current_scene(params);
    if (method_name == "set_debug_collisions_hint") return handler.handle_set_debug_collisions_hint(params);
    if (method_name == "set_debug_navigation_hint") return handler.handle_set_debug_navigation_hint(params);
    if (method_name == "set_debug_paths_hint") return handler.handle_set_debug_paths_hint(params);
    if (method_name == "set_deferred") return handler.handle_set_deferred(params);
    if (method_name == "set_edited_scene_root") return handler.handle_set_edited_scene_root(params);
    if (method_name == "set_group") return handler.handle_set_group(params);
    if (method_name == "set_group_flags") return handler.handle_set_group_flags(params);
    if (method_name == "set_indexed") return handler.handle_set_indexed(params);
    if (method_name == "set_message_translation") return handler.handle_set_message_translation(params);
    if (method_name == "set_meta") return handler.handle_set_meta(params);
    if (method_name == "set_multiplayer") return handler.handle_set_multiplayer(params);
    if (method_name == "set_multiplayer_poll_enabled") return handler.handle_set_multiplayer_poll_enabled(params);
    if (method_name == "set_pause") return handler.handle_set_pause(params);
    if (method_name == "set_physics_interpolation_enabled") return handler.handle_set_physics_interpolation_enabled(params);
    if (method_name == "set_quit_on_go_back") return handler.handle_set_quit_on_go_back(params);
    if (method_name == "set_script") return handler.handle_set_script(params);
    if (method_name == "set_translation_domain") return handler.handle_set_translation_domain(params);
    if (method_name == "to_string") return handler.handle_to_string(params);
    if (method_name == "tr") return handler.handle_tr(params);
    if (method_name == "tr_n") return handler.handle_tr_n(params);
    if (method_name == "unload_current_scene") return handler.handle_unload_current_scene(params);

    // Method not found
    Dictionary error_response;
    Dictionary error;
    error["code"] = -32601;
    error["message"] = "Method not found";
    error_response["jsonrpc"] = "2.0";
    error_response["error"] = error;
    return error_response;
}

// VM entrypoint wrapper for JSON-RPC dispatcher (callv ABI)
PUBLIC Dictionary jsonrpc_dispatch(const Array& callv_args) {
    Variant request_v = callv_args[0];
    if (request_v.get_type() != Variant::DICTIONARY) {
        Dictionary error_response = Dictionary::Create();
        Dictionary error = Dictionary::Create();
        error["code"] = -32602;
        error["message"] = "Invalid params";
        error_response["jsonrpc"] = "2.0";
        error_response["error"] = error;
        error_response["id"] = Variant();
        return error_response;
    }

    return jsonrpc_dispatch_impl(Dictionary(request_v));
}

static Object instantiate_engine_class(const String& class_name) {
    Object class_db("ClassDB");
    if (!class_db.is_valid()) {
        return Object(0);
    }
    Variant created = class_db.call("instantiate", class_name);
    if (created.get_type() != Variant::OBJECT) {
        return Object(0);
    }
    return Object(created);
}

static Dictionary make_jsonrpc_error(int code, const String& message, const Variant& id) {
    Dictionary response = Dictionary::Create();
    Dictionary error = Dictionary::Create();
    error["code"] = code;
    error["message"] = message;
    response["jsonrpc"] = "2.0";
    response["error"] = error;
    response["id"] = id;
    return response;
}

static Dictionary make_jsonrpc_error_no_id(int code, const String& message) {
    return make_jsonrpc_error(code, message, Variant());
}

// Standalone entry point - initializes Godot and starts JSON-RPC server
PUBLIC int standalone_main() {
    print("Standalone scenetree_passthrough starting...");
    
    // Initialize Godot engine components manually
    Object godot("Engine");
    if (godot.is_valid()) {
        godot.call("_init");
        godot.call("set_data_path", "/tmp/godot_standalone_data");
    }
    
    // Setup SceneTree
    Object scene_tree = instantiate_engine_class("SceneTree");
    if (!scene_tree.is_valid()) {
        print("ERROR: Failed to create SceneTree");
        return 1;
    }
    
    // Create root node
    Object root_node = instantiate_engine_class("Node");
    if (!root_node.is_valid()) {
        print("ERROR: Failed to create root Node");
        return 1;
    }
    root_node.call("set_name", "root");
    scene_tree.call("set_root", root_node);
    
    print("Godot engine initialized with SceneTree");
    
    // Add direct API functions
    ADD_API_FUNCTION(jsonrpc_dispatch, "Dictionary", "Array callv_args", 
        "Routes JSON-RPC 2.0 requests to SceneTree and Object methods");
    
    // Add individual API functions
    ADD_API_FUNCTION(_initialize, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_physics_process, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_process, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_finalize, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_init, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_to_string, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_notification, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_set, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_get, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_get_property_list, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_validate_property, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_property_can_revert, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_property_get_revert, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_init, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_next, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_get, "Dictionary", "Array callv_args", "Virtual API binding");
    
    ADD_API_FUNCTION(start_jsonrpc_server, "Dictionary", "Array callv_args", 
        "Start TCP JSON-RPC 2.0 server (params[0]=port, default 7777)");
    
    // Start JSON-RPC server on port 7777
    Array start_args = Array::Create();
    start_args.push_back(7777);
    Dictionary startup = start_jsonrpc_server(start_args);
    if (startup.has("error")) {
        print("Failed to start JSON-RPC TCP server");
        return 1;
    }
    
    Dictionary ok = startup["result"];
    print("JSON-RPC TCP server started on " + String(ok.get("host")) + ":" + String(ok.get("port")));
    
    print("Registered " + String(Array(start_args).size()) + " virtual API bindings");
    print("Standalone scenetree_passthrough running. Ready to accept connections.");
    
    // Enter main loop - poll JSON-RPC server
    while (true) {
        // Process any pending JSON-RPC connections
        poll_jsonrpc_server(Object(0));
        OS::get_singleton()->delay_msec(10);
    }
    
    return 0;
}

// Fallback main for compatibility
int main() {
#if defined(STANDALONE_NATIVE)
    return standalone_main();
#else
    // Original scenetree_passthrough behavior
    print("JSON-RPC SceneTree Server initialized");
    
    ADD_API_FUNCTION(jsonrpc_dispatch, "Dictionary", "Array callv_args", 
        "Routes JSON-RPC 2.0 requests to SceneTree and Object methods");
    
    ADD_API_FUNCTION(_initialize, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_physics_process, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_process, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_finalize, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_init, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_to_string, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_notification, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_set, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_get, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_get_property_list, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_validate_property, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_property_can_revert, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_property_get_revert, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_init, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_next, "Dictionary", "Array callv_args", "Virtual API binding");
    ADD_API_FUNCTION(_iter_get, "Dictionary", "Array callv_args", "Virtual API binding");
    
    ADD_API_FUNCTION(start_jsonrpc_server, "Dictionary", "Array callv_args", 
        "Start TCP JSON-RPC 2.0 server (params[0]=port, default 7777)");
    
    Array start_args = Array::Create();
    start_args.push_back(7777);
    Dictionary startup = start_jsonrpc_server(start_args);
    if (startup.has("error")) {
        print("Failed to start JSON-RPC TCP server");
        return 1;
    } else {
        print("JSON-RPC TCP server auto-started on 127.0.0.1:7777");
    }
    
    print("Registered virtual API bindings from programs/scenetree_passthrough/api/virtual_methods_v4.6.1_stable.jsonl");
    
    halt();
    return 0;
#endif
}

static Variant poll_jsonrpc_server(Object) {
    if (!jsonrpc_server_started || !jsonrpc_tcp_server.is_valid()) {
        return Variant();
    }

    while (bool(jsonrpc_tcp_server.call("is_connection_available"))) {
        Variant peer_v = jsonrpc_tcp_server.call("take_connection");
        if (peer_v.get_type() == Variant::OBJECT) {
            Object peer(peer_v);
            if (peer.is_valid()) {
                jsonrpc_clients.push_back(peer);
            }
        } else {
            break;
        }
    }

    std::vector<Object> alive;
    alive.reserve(jsonrpc_clients.size());

    for (Object& peer : jsonrpc_clients) {
        if (!peer.is_valid()) {
            continue;
        }

        peer.call("poll");

        int status = int(peer.call("get_status"));
        if (status != 2 && status != 5) {
            continue;
        }

        int available = int(peer.call("get_available_bytes"));
        if (available > 0) {
            String request_text = String(peer.call("get_utf8_string", available));
            String response_text = handle_jsonrpc_message_text(request_text);
            if (!response_text.is_empty()) {
                peer.call("put_utf8_string", response_text);
            }
        }

        alive.push_back(peer);
    }

    jsonrpc_clients = std::move(alive);
    return Variant();
}

PUBLIC Dictionary start_jsonrpc_server(const Array& callv_args) {
    int port = int(callv_args[0]);

    Object server = instantiate_engine_class("TCPServer");
    if (!server.is_valid()) {
        Dictionary error = Dictionary::Create();
        error["jsonrpc"] = "2.0";
        Dictionary body = Dictionary::Create();
        body["code"] = -32000;
        body["message"] = "Unable to instantiate TCPServer";
        error["error"] = body;
        error["id"] = Variant();
        return error;
    }

    Variant listen_result = server.call("listen", port, String("127.0.0.1"));
    if (int(listen_result) != 0) {
        Dictionary error = Dictionary::Create();
        error["jsonrpc"] = "2.0";
        Dictionary body = Dictionary::Create();
        body["code"] = -32000;
        body["message"] = "TCP listen failed";
        body["data"] = listen_result;
        error["error"] = body;
        error["id"] = Variant();
        return error;
    }

    jsonrpc_tcp_server = server;
    jsonrpc_server_port = port;
    jsonrpc_server_started = true;
    CallbackTimer::periodic(0.01, poll_jsonrpc_server);

    Dictionary ok = Dictionary::Create();
    Dictionary result = Dictionary::Create();
    result["host"] = "127.0.0.1";
    result["port"] = port;
    ok["jsonrpc"] = "2.0";
    ok["result"] = result;
    ok["id"] = Variant();
    return ok;
}

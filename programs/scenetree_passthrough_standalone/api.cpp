#include "api.hpp"
#include "handlers.hpp"

// API function table for standalone mode
// These functions are exported via callv ABI for JSON-RPC dispatch

static JSONRPCHandler handler;

// JSON-RPC 2.0 dispatcher entry point (callv ABI)
PUBLIC Dictionary jsonrpc_dispatch(const Array& callv_args) {
    if (callv_args.size() < 1) {
        Dictionary error;
        error["jsonrpc"] = "2.0";
        Dictionary body;
        body["code"] = -32602;
        body["message"] = "Invalid params";
        error["error"] = body;
        return error;
    }

    Variant request = callv_args[0];
    if (request.get_type() != Variant::DICTIONARY) {
        Dictionary error;
        error["jsonrpc"] = "2.0";
        Dictionary body;
        body["code"] = -32600;
        body["message"] = "Invalid request";
        error["error"] = body;
        return error;
    }

    Dictionary request_dict = Dictionary(request);
    String method = request_dict.get("method", String());
    Array params = request_dict.get("params", Array());
    Variant id = request_dict.get("id", Variant());

    // Route to handler
    if (method == "get_root") return handler.handle_get_root(params);
    if (method == "has_group") return handler.handle_has_group(params);
    if (method == "is_accessibility_enabled") return handler.handle_is_accessibility_enabled(params);
    if (method == "is_accessibility_supported") return handler.handle_is_accessibility_supported(params);
    if (method == "set_auto_accept_quit") return handler.handle_set_auto_accept_quit(params);
    if (method == "is_auto_accept_quit") return handler.handle_is_auto_accept_quit(params);
    if (method == "set_quit_on_go_back") return handler.handle_set_quit_on_go_back(params);
    if (method == "is_quit_on_go_back") return handler.handle_is_quit_on_go_back(params);
    if (method == "set_debug_collisions_hint") return handler.handle_set_debug_collisions_hint(params);
    if (method == "is_debugging_collisions_hint") return handler.handle_is_debugging_collisions_hint(params);
    if (method == "set_debug_paths_hint") return handler.handle_set_debug_paths_hint(params);
    if (method == "is_debugging_paths_hint") return handler.handle_is_debugging_paths_hint(params);
    if (method == "set_debug_navigation_hint") return handler.handle_set_debug_navigation_hint(params);
    if (method == "is_debugging_navigation_hint") return handler.handle_is_debugging_navigation_hint(params);
    if (method == "set_edited_scene_root") return handler.handle_set_edited_scene_root(params);
    if (method == "get_edited_scene_root") return handler.handle_get_edited_scene_root(params);
    if (method == "set_pause") return handler.handle_set_pause(params);
    if (method == "is_paused") return handler.handle_is_paused(params);
    if (method == "create_timer") return handler.handle_create_timer(params);
    if (method == "create_tween") return handler.handle_create_tween(params);
    if (method == "get_processed_tweens") return handler.handle_get_processed_tweens(params);
    if (method == "get_node_count") return handler.handle_get_node_count(params);
    if (method == "get_frame") return handler.handle_get_frame(params);
    if (method == "quit") return handler.handle_quit(params);
    if (method == "set_physics_interpolation_enabled") return handler.handle_set_physics_interpolation_enabled(params);
    if (method == "is_physics_interpolation_enabled") return handler.handle_is_physics_interpolation_enabled(params);
    if (method == "queue_delete") return handler.handle_queue_delete(params);
    if (method == "call_group") return handler.handle_call_group(params);
    if (method == "call_group_flags") return handler.handle_call_group_flags(params);
    if (method == "notify_group_flags") return handler.handle_notify_group_flags(params);
    if (method == "set_group_flags") return handler.handle_set_group_flags(params);
    if (method == "notify_group") return handler.handle_notify_group(params);
    if (method == "set_group") return handler.handle_set_group(params);
    if (method == "get_nodes_in_group") return handler.handle_get_nodes_in_group(params);
    if (method == "get_first_node_in_group") return handler.handle_get_first_node_in_group(params);
    if (method == "get_node_count_in_group") return handler.handle_get_node_count_in_group(params);
    if (method == "set_current_scene") return handler.handle_set_current_scene(params);
    if (method == "get_current_scene") return handler.handle_get_current_scene(params);
    if (method == "change_scene_to_file") return handler.handle_change_scene_to_file(params);
    if (method == "change_scene_to_packed") return handler.handle_change_scene_to_packed(params);
    if (method == "change_scene_to_node") return handler.handle_change_scene_to_node(params);
    if (method == "reload_current_scene") return handler.handle_reload_current_scene(params);
    if (method == "unload_current_scene") return handler.handle_unload_current_scene(params);
    if (method == "set_multiplayer") return handler.handle_set_multiplayer(params);
    if (method == "get_multiplayer") return handler.handle_get_multiplayer(params);
    if (method == "set_multiplayer_poll_enabled") return handler.handle_set_multiplayer_poll_enabled(params);
    if (method == "is_multiplayer_poll_enabled") return handler.handle_is_multiplayer_poll_enabled(params);
    if (method == "free") return handler.handle_free(params);
    if (method == "get_class") return handler.handle_get_class(params);
    if (method == "is_class") return handler.handle_is_class(params);
    if (method == "set") return handler.handle_set(params);
    if (method == "get") return handler.handle_get(params);
    if (method == "set_indexed") return handler.handle_set_indexed(params);
    if (method == "get_indexed") return handler.handle_get_indexed(params);
    if (method == "get_property_list") return handler.handle_get_property_list(params);
    if (method == "get_method_list") return handler.handle_get_method_list(params);
    if (method == "property_can_revert") return handler.handle_property_can_revert(params);
    if (method == "property_get_revert") return handler.handle_property_get_revert(params);
    if (method == "notification") return handler.handle_notification(params);
    if (method == "to_string") return handler.handle_to_string(params);
    if (method == "get_instance_id") return handler.handle_get_instance_id(params);
    if (method == "set_script") return handler.handle_set_script(params);
    if (method == "get_script") return handler.handle_get_script(params);
    if (method == "set_meta") return handler.handle_set_meta(params);
    if (method == "remove_meta") return handler.handle_remove_meta(params);
    if (method == "get_meta") return handler.handle_get_meta(params);
    if (method == "has_meta") return handler.handle_has_meta(params);
    if (method == "get_meta_list") return handler.handle_get_meta_list(params);
    if (method == "add_user_signal") return handler.handle_add_user_signal(params);
    if (method == "has_user_signal") return handler.handle_has_user_signal(params);
    if (method == "remove_user_signal") return handler.handle_remove_user_signal(params);
    if (method == "emit_signal") return handler.handle_emit_signal(params);
    if (method == "call") return handler.handle_call(params);
    if (method == "call_deferred") return handler.handle_call_deferred(params);
    if (method == "set_deferred") return handler.handle_set_deferred(params);
    if (method == "callv") return handler.handle_callv(params);
    if (method == "has_method") return handler.handle_has_method(params);
    if (method == "get_method_argument_count") return handler.handle_get_method_argument_count(params);
    if (method == "has_signal") return handler.handle_has_signal(params);
    if (method == "get_signal_list") return handler.handle_get_signal_list(params);
    if (method == "get_signal_connection_list") return handler.handle_get_signal_connection_list(params);
    if (method == "get_incoming_connections") return handler.handle_get_incoming_connections(params);
    if (method == "connect") return handler.handle_connect(params);
    if (method == "disconnect") return handler.handle_disconnect(params);
    if (method == "is_connected") return handler.handle_is_connected(params);
    if (method == "has_connections") return handler.handle_has_connections(params);
    if (method == "set_block_signals") return handler.handle_set_block_signals(params);
    if (method == "is_blocking_signals") return handler.handle_is_blocking_signals(params);
    if (method == "notify_property_list_changed") return handler.handle_notify_property_list_changed(params);
    if (method == "set_message_translation") return handler.handle_set_message_translation(params);
    if (method == "can_translate_messages") return handler.handle_can_translate_messages(params);
    if (method == "tr") return handler.handle_tr(params);
    if (method == "tr_n") return handler.handle_tr_n(params);
    if (method == "get_translation_domain") return handler.handle_get_translation_domain(params);
    if (method == "set_translation_domain") return handler.handle_set_translation_domain(params);
    if (method == "is_queued_for_deletion") return handler.handle_is_queued_for_deletion(params);
    if (method == "cancel_free") return handler.handle_cancel_free(params);
    if (method == "_initialize") return handler.handle__initialize(params);
    if (method == "_physics_process") return handler.handle__physics_process(params);
    if (method == "_process") return handler.handle__process(params);
    if (method == "_finalize") return handler.handle__finalize(params);
    if (method == "_init") return handler.handle__init(params);
    if (method == "_to_string") return handler.handle__to_string(params);
    if (method == "_notification") return handler.handle__notification(params);
    if (method == "_set") return handler.handle__set(params);
    if (method == "_get") return handler.handle__get(params);
    if (method == "_get_property_list") return handler.handle__get_property_list(params);
    if (method == "_validate_property") return handler.handle__validate_property(params);
    if (method == "_property_can_revert") return handler.handle__property_can_revert(params);
    if (method == "_property_get_revert") return handler.handle__property_get_revert(params);
    if (method == "_iter_init") return handler.handle__iter_init(params);
    if (method == "_iter_next") return handler.handle__iter_next(params);
    if (method == "_iter_get") return handler.handle__iter_get(params);

    // Method not found
    Dictionary error;
    error["jsonrpc"] = "2.0";
    Dictionary body;
    body["code"] = -32601;
    body["message"] = "Method not found";
    error["error"] = body;
    error["id"] = id;
    return error;
}

// Start the JSON-RPC TCP server
PUBLIC Dictionary start_jsonrpc_server(const Array& callv_args) {
    int port = 7777;
    if (callv_args.size() > 0) {
        port = int(callv_args[0]);
    }

    Object tcp_server = Object("TCPServer");
    if (!tcp_server.is_valid()) {
        Dictionary error;
        error["jsonrpc"] = "2.0";
        Dictionary body;
        body["code"] = -32000;
        body["message"] = "Unable to instantiate TCPServer";
        error["error"] = body;
        error["id"] = Variant();
        return error;
    }

    // Initialize the server
    tcp_server.call("_init");

    // Try to bind to port
    Variant result = tcp_server.call("listen", port, String("0.0.0.0"));
    if (int(result) != 0) {
        Dictionary error;
        error["jsonrpc"] = "2.0";
        Dictionary body;
        body["code"] = -32000;
        body["message"] = "TCP listen failed";
        body["data"] = result;
        error["error"] = body;
        error["id"] = Variant();
        return error;
    }

    Dictionary ok;
    ok["jsonrpc"] = "2.0";
    Dictionary result_dict;
    result_dict["host"] = "0.0.0.0";
    result_dict["port"] = port;
    ok["result"] = result_dict;
    ok["id"] = Variant();

    return ok;
}

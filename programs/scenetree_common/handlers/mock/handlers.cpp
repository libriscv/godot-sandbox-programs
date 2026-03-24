#include "handlers.hpp"

#include <climits>
#include <vector>

// ==================== Utility Methods ====================

static bool is_valid_array(const Array& params) {
    const unsigned idx = params.get_variant_index();
    return idx != static_cast<unsigned>(INT32_MIN);
}

static int safe_array_size(const Array& params) {
    return is_valid_array(params) ? params.size() : 0;
}

static bool has_explicit_target_selector(const Array& args) {
    if (safe_array_size(args) == 0) {
        return false;
    }

    Variant first = args[0];
    if (first.get_type() == Variant::OBJECT) {
        return true;
    }
    if (first.get_type() == Variant::STRING) {
        String path = String(first);
        return path.begins_with("/");
    }
    return false;
}

static std::vector<Variant> to_variant_vector(const Array& params, int start = 0) {
    std::vector<Variant> argv;
    const int size = safe_array_size(params);
    if (start < 0 || start > size) {
        return argv;
    }
    argv.reserve(size - start);
    for (int i = start; i < size; ++i) {
        argv.push_back(params.at(i));
    }
    return argv;
}

Dictionary JSONRPCHandler::create_ok(const Variant& result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok(const String& result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok(int result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok(bool result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok(const Array& result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok(const Dictionary& result) {
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["result"] = result;
    return response;
}

Dictionary JSONRPCHandler::create_ok_string(const String& message) {
    return create_ok(message);
}

Dictionary JSONRPCHandler::create_error(int code, const String& message, const String& data) {
    Dictionary error = Dictionary::Create();
    error["code"] = code;
    error["message"] = message;
    if (!data.is_empty()) {
        error["data"] = data;
    }
    
    Dictionary response = Dictionary::Create();
    response["jsonrpc"] = "2.0";
    response["error"] = error;
    return response;
}

Object JSONRPCHandler::get_node_or_object(const Array& args) {
    Node self = get_node<Node>(".");
    if (!self.is_valid()) {
        return Object(0);
    }
    Object self_obj(self.address());

    if (safe_array_size(args) > 0) {
        Variant first = args[0];
        if (first.get_type() == Variant::OBJECT) {
            return Object(first);
        }
        if (first.get_type() == Variant::STRING) {
            String path = String(first);
            if (!path.begins_with("/")) {
                goto fallback_current;
            }
            Variant argv[] = { Variant(path) };
            Variant node = self_obj.callv("get_node", false, argv, 1);
            if (node.get_type() == Variant::OBJECT) {
                return Object(node);
            }
            return Object(0);
        }
    }

fallback_current:
    Variant current_argv[] = { Variant(String(".")) };
    Variant current = self_obj.callv("get_node", false, current_argv, 1);
    if (current.get_type() == Variant::OBJECT) {
        return Object(current);
    }
    return self_obj;
}

// ==================== SceneTree Methods ====================

Dictionary JSONRPCHandler::handle_get_root(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("get_root", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_has_group(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    if (safe_array_size(params) == 0) return create_error(-32000, "Missing group name", "");
    std::vector<Variant> argv = to_variant_vector(params);
    return create_ok(tree.callv("has_group", false, argv.data(), (unsigned)argv.size()));
}

Dictionary JSONRPCHandler::handle_is_accessibility_enabled(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("is_accessibility_enabled", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_is_accessibility_supported(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("is_accessibility_supported", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_set_pause(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    if (safe_array_size(params) == 0) return create_error(-32000, "Missing pause value", "");
    std::vector<Variant> argv = to_variant_vector(params);
    return create_ok(tree.callv("set_pause", false, argv.data(), (unsigned)argv.size()));
}

Dictionary JSONRPCHandler::handle_is_paused(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("is_paused", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_quit(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    std::vector<Variant> argv = to_variant_vector(params);
    if (argv.empty()) {
        argv.push_back(Variant(0));
    }
    return create_ok(tree.callv("quit", false, argv.data(), (unsigned)argv.size()));
}

Dictionary JSONRPCHandler::handle_reload_current_scene(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("reload_current_scene", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_change_scene_to_file(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    if (safe_array_size(params) == 0) return create_error(-32000, "Missing scene path", "");
    std::vector<Variant> argv = to_variant_vector(params);
    return create_ok(tree.callv("change_scene_to_file", false, argv.data(), (unsigned)argv.size()));
}

Dictionary JSONRPCHandler::handle_call_group(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    std::vector<Variant> argv = to_variant_vector(params);
    Variant result = tree.callv("call_group", false, argv.data(), (unsigned)argv.size());
    return create_ok(result);
}

Dictionary JSONRPCHandler::handle_queue_delete(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");

    Object target = get_node_or_object(params);
    if (!target.is_valid()) {
        return create_error(-32000, "Target object unavailable", "queue_delete");
    }

    Variant argv[] = { Variant(target) };
    Variant result = tree.callv("queue_delete", false, argv, 1);
    return create_ok(result);
}

// Stub implementations for other SceneTree methods
Dictionary JSONRPCHandler::handle_is_auto_accept_quit(const Array& params) {
    (void)params;
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    return create_ok(tree.callv("is_auto_accept_quit", false, nullptr, 0));
}

Dictionary JSONRPCHandler::handle_set_auto_accept_quit(const Array& params) {
    SceneTree tree = get_tree();
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", "");
    if (safe_array_size(params) == 0) return create_error(-32000, "Missing value", "");
    std::vector<Variant> argv = to_variant_vector(params);
    return create_ok(tree.callv("set_auto_accept_quit", false, argv.data(), (unsigned)argv.size()));
}

// Placeholder implementations for remaining methods
#define STUB_SCENETREE(name, arity) Dictionary JSONRPCHandler::handle_##name(const Array& params) { \
    SceneTree tree = get_tree(); \
    if (!tree.is_valid()) return create_error(-32000, "SceneTree unavailable", ""); \
    constexpr int expected_arity = arity; \
    (void)expected_arity; \
    std::vector<Variant> argv = to_variant_vector(params); \
    Variant result = tree.callv(#name, false, argv.data(), (unsigned)argv.size()); \
    return create_ok(result); \
}

STUB_SCENETREE(is_quit_on_go_back, 0)
STUB_SCENETREE(set_quit_on_go_back, 1)
STUB_SCENETREE(set_debug_collisions_hint, 1)
STUB_SCENETREE(is_debugging_collisions_hint, 0)
STUB_SCENETREE(set_debug_paths_hint, 1)
STUB_SCENETREE(is_debugging_paths_hint, 0)
STUB_SCENETREE(set_debug_navigation_hint, 1)
STUB_SCENETREE(is_debugging_navigation_hint, 0)
STUB_SCENETREE(_initialize, 0)
STUB_SCENETREE(_physics_process, 1)
STUB_SCENETREE(_process, 1)
STUB_SCENETREE(_finalize, 0)
STUB_SCENETREE(set_edited_scene_root, 1)
STUB_SCENETREE(get_edited_scene_root, 0)
STUB_SCENETREE(create_timer, -1)
STUB_SCENETREE(create_tween, 0)
STUB_SCENETREE(get_processed_tweens, 0)
STUB_SCENETREE(get_node_count, 0)
STUB_SCENETREE(get_frame, 0)
STUB_SCENETREE(set_physics_interpolation_enabled, 1)
STUB_SCENETREE(is_physics_interpolation_enabled, 0)
STUB_SCENETREE(call_group_flags, -1)
STUB_SCENETREE(notify_group_flags, -1)
STUB_SCENETREE(set_group_flags, -1)
STUB_SCENETREE(notify_group, -1)
STUB_SCENETREE(set_group, -1)
STUB_SCENETREE(get_nodes_in_group, 1)
STUB_SCENETREE(get_first_node_in_group, 1)
STUB_SCENETREE(get_node_count_in_group, 1)
STUB_SCENETREE(set_current_scene, 1)
STUB_SCENETREE(get_current_scene, 0)
STUB_SCENETREE(change_scene_to_packed, 1)
STUB_SCENETREE(change_scene_to_node, 1)
STUB_SCENETREE(unload_current_scene, 0)
STUB_SCENETREE(set_multiplayer, -1)
STUB_SCENETREE(get_multiplayer, -1)
STUB_SCENETREE(set_multiplayer_poll_enabled, 1)
STUB_SCENETREE(is_multiplayer_poll_enabled, 0)

// ==================== Object/Node Methods ====================

#define STUB_OBJECT(name, arity) Dictionary JSONRPCHandler::handle_##name(const Array& params) { \
    Object target = get_node_or_object(params); \
    if (!target.is_valid()) return create_error(-32000, "Target object unavailable", #name); \
    constexpr int expected_arity = arity; \
    (void)expected_arity; \
    const int start = has_explicit_target_selector(params) ? 1 : 0; \
    std::vector<Variant> argv = to_variant_vector(params, start); \
    Variant result = target.callv(#name, false, argv.data(), (unsigned)argv.size()); \
    return create_ok(result); \
}

STUB_OBJECT(free, 0)
STUB_OBJECT(get_class, 0)
STUB_OBJECT(is_class, 1)
STUB_OBJECT(set, 2)
STUB_OBJECT(get, 1)
STUB_OBJECT(set_indexed, 2)
STUB_OBJECT(get_indexed, 1)
STUB_OBJECT(get_property_list, 0)
STUB_OBJECT(get_method_list, 0)
STUB_OBJECT(property_can_revert, 1)
STUB_OBJECT(property_get_revert, 1)
STUB_OBJECT(notification, -1)
STUB_OBJECT(to_string, 0)
STUB_OBJECT(get_instance_id, 0)
STUB_OBJECT(set_script, 1)
STUB_OBJECT(get_script, 0)
STUB_OBJECT(set_meta, 2)
STUB_OBJECT(remove_meta, 1)
STUB_OBJECT(get_meta, -1)
STUB_OBJECT(has_meta, 1)
STUB_OBJECT(get_meta_list, 0)
STUB_OBJECT(add_user_signal, 1)
STUB_OBJECT(has_user_signal, 1)
STUB_OBJECT(remove_user_signal, 1)
STUB_OBJECT(emit_signal, -1)
STUB_OBJECT(call, -1)
STUB_OBJECT(call_deferred, -1)
STUB_OBJECT(set_deferred, 2)
STUB_OBJECT(callv, 2)
STUB_OBJECT(has_method, 1)
STUB_OBJECT(get_method_argument_count, 1)
STUB_OBJECT(has_signal, 1)
STUB_OBJECT(get_signal_list, 0)
STUB_OBJECT(get_signal_connection_list, 1)
STUB_OBJECT(get_incoming_connections, 0)
STUB_OBJECT(connect, -1)
STUB_OBJECT(disconnect, 2)
STUB_OBJECT(is_connected, 2)
STUB_OBJECT(has_connections, 1)
STUB_OBJECT(set_block_signals, 1)
STUB_OBJECT(is_blocking_signals, 0)
STUB_OBJECT(notify_property_list_changed, 0)
STUB_OBJECT(set_message_translation, 1)
STUB_OBJECT(can_translate_messages, 0)
STUB_OBJECT(tr, -1)
STUB_OBJECT(tr_n, -1)
STUB_OBJECT(get_translation_domain, 0)
STUB_OBJECT(set_translation_domain, 1)
STUB_OBJECT(is_queued_for_deletion, 0)
STUB_OBJECT(cancel_free, 0)
STUB_OBJECT(_init, 0)
STUB_OBJECT(_to_string, 0)
STUB_OBJECT(_notification, 1)
STUB_OBJECT(_set, 2)
STUB_OBJECT(_get, 1)
STUB_OBJECT(_get_property_list, 0)
STUB_OBJECT(_validate_property, 1)
STUB_OBJECT(_property_can_revert, 1)
STUB_OBJECT(_property_get_revert, 1)
STUB_OBJECT(_iter_init, 1)
STUB_OBJECT(_iter_next, 1)
STUB_OBJECT(_iter_get, 1)


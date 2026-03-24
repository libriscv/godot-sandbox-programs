#pragma once

#include <api.hpp>
#include <string>
#include <unordered_map>
#include <functional>

// ==================== JSONRPCHandler ====================
// Handles JSON-RPC 2.0 requests for SceneTree and Object method calls
// Uses Dictionary/Array instead of Variant for type safety

class JSONRPCHandler {
private:
    // Utility methods for JSON-RPC responses
    static Dictionary create_ok(const Variant& result);
    static Dictionary create_ok(const String& result);
    static Dictionary create_ok(int result);
    static Dictionary create_ok(bool result);
    static Dictionary create_ok(const Array& result);
    static Dictionary create_ok(const Dictionary& result);
    static Dictionary create_ok_string(const String& message);
    static Dictionary create_error(int code, const String& message, const String& data = "");
    
    // Utility to get a node or object from a node path string
    static Object get_node_or_object(const Array& args);

public:
    // MainLoop Methods
    Dictionary handle__initialize(const Array& params);
    Dictionary handle__physics_process(const Array& params);
    Dictionary handle__process(const Array& params);
    Dictionary handle__finalize(const Array& params);
    Dictionary handle__init(const Array& params);
    Dictionary handle__to_string(const Array& params);
    Dictionary handle__notification(const Array& params);
    Dictionary handle__set(const Array& params);
    Dictionary handle__get(const Array& params);
    Dictionary handle__get_property_list(const Array& params);
    Dictionary handle__validate_property(const Array& params);
    Dictionary handle__property_can_revert(const Array& params);
    Dictionary handle__property_get_revert(const Array& params);
    Dictionary handle__iter_init(const Array& params);
    Dictionary handle__iter_next(const Array& params);
    Dictionary handle__iter_get(const Array& params);

    // SceneTree Methods
    Dictionary handle_get_root(const Array& params);
    Dictionary handle_has_group(const Array& params);
    Dictionary handle_is_accessibility_enabled(const Array& params);
    Dictionary handle_is_accessibility_supported(const Array& params);
    Dictionary handle_is_auto_accept_quit(const Array& params);
    Dictionary handle_set_auto_accept_quit(const Array& params);
    Dictionary handle_is_quit_on_go_back(const Array& params);
    Dictionary handle_set_quit_on_go_back(const Array& params);
    Dictionary handle_set_debug_collisions_hint(const Array& params);
    Dictionary handle_is_debugging_collisions_hint(const Array& params);
    Dictionary handle_set_debug_paths_hint(const Array& params);
    Dictionary handle_is_debugging_paths_hint(const Array& params);
    Dictionary handle_set_debug_navigation_hint(const Array& params);
    Dictionary handle_is_debugging_navigation_hint(const Array& params);
    Dictionary handle_set_edited_scene_root(const Array& params);
    Dictionary handle_get_edited_scene_root(const Array& params);
    Dictionary handle_set_pause(const Array& params);
    Dictionary handle_is_paused(const Array& params);
    Dictionary handle_create_timer(const Array& params);
    Dictionary handle_create_tween(const Array& params);
    Dictionary handle_get_processed_tweens(const Array& params);
    Dictionary handle_get_node_count(const Array& params);
    Dictionary handle_get_frame(const Array& params);
    Dictionary handle_quit(const Array& params);
    Dictionary handle_set_physics_interpolation_enabled(const Array& params);
    Dictionary handle_is_physics_interpolation_enabled(const Array& params);
    Dictionary handle_queue_delete(const Array& params);
    Dictionary handle_call_group_flags(const Array& params);
    Dictionary handle_notify_group_flags(const Array& params);
    Dictionary handle_set_group_flags(const Array& params);
    Dictionary handle_call_group(const Array& params);
    Dictionary handle_notify_group(const Array& params);
    Dictionary handle_set_group(const Array& params);
    Dictionary handle_get_nodes_in_group(const Array& params);
    Dictionary handle_get_first_node_in_group(const Array& params);
    Dictionary handle_get_node_count_in_group(const Array& params);
    Dictionary handle_set_current_scene(const Array& params);
    Dictionary handle_get_current_scene(const Array& params);
    Dictionary handle_change_scene_to_file(const Array& params);
    Dictionary handle_change_scene_to_packed(const Array& params);
    Dictionary handle_change_scene_to_node(const Array& params);
    Dictionary handle_reload_current_scene(const Array& params);
    Dictionary handle_unload_current_scene(const Array& params);
    Dictionary handle_set_multiplayer(const Array& params);
    Dictionary handle_get_multiplayer(const Array& params);
    Dictionary handle_set_multiplayer_poll_enabled(const Array& params);
    Dictionary handle_is_multiplayer_poll_enabled(const Array& params);
    
    // Object/Node Methods
    Dictionary handle_free(const Array& params);
    Dictionary handle_get_class(const Array& params);
    Dictionary handle_is_class(const Array& params);
    Dictionary handle_set(const Array& params);
    Dictionary handle_get(const Array& params);
    Dictionary handle_set_indexed(const Array& params);
    Dictionary handle_get_indexed(const Array& params);
    Dictionary handle_get_property_list(const Array& params);
    Dictionary handle_get_method_list(const Array& params);
    Dictionary handle_property_can_revert(const Array& params);
    Dictionary handle_property_get_revert(const Array& params);
    Dictionary handle_notification(const Array& params);
    Dictionary handle_to_string(const Array& params);
    Dictionary handle_get_instance_id(const Array& params);
    Dictionary handle_set_script(const Array& params);
    Dictionary handle_get_script(const Array& params);
    Dictionary handle_set_meta(const Array& params);
    Dictionary handle_remove_meta(const Array& params);
    Dictionary handle_get_meta(const Array& params);
    Dictionary handle_has_meta(const Array& params);
    Dictionary handle_get_meta_list(const Array& params);
    Dictionary handle_add_user_signal(const Array& params);
    Dictionary handle_has_user_signal(const Array& params);
    Dictionary handle_remove_user_signal(const Array& params);
    Dictionary handle_emit_signal(const Array& params);
    Dictionary handle_call(const Array& params);
    Dictionary handle_call_deferred(const Array& params);
    Dictionary handle_set_deferred(const Array& params);
    Dictionary handle_callv(const Array& params);
    Dictionary handle_has_method(const Array& params);
    Dictionary handle_get_method_argument_count(const Array& params);
    Dictionary handle_has_signal(const Array& params);
    Dictionary handle_get_signal_list(const Array& params);
    Dictionary handle_get_signal_connection_list(const Array& params);
    Dictionary handle_get_incoming_connections(const Array& params);
    Dictionary handle_connect(const Array& params);
    Dictionary handle_disconnect(const Array& params);
    Dictionary handle_is_connected(const Array& params);
    Dictionary handle_has_connections(const Array& params);
    Dictionary handle_set_block_signals(const Array& params);
    Dictionary handle_is_blocking_signals(const Array& params);
    Dictionary handle_notify_property_list_changed(const Array& params);
    Dictionary handle_set_message_translation(const Array& params);
    Dictionary handle_can_translate_messages(const Array& params);
    Dictionary handle_tr(const Array& params);
    Dictionary handle_tr_n(const Array& params);
    Dictionary handle_get_translation_domain(const Array& params);
    Dictionary handle_set_translation_domain(const Array& params);
    Dictionary handle_is_queued_for_deletion(const Array& params);
    Dictionary handle_cancel_free(const Array& params);
};

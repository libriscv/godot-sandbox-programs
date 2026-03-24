# SceneTree Passthrough API Reference
## Overview
This API exposes Godot's SceneTree and Node methods via JSON-RPC 2.0. All method names match the Godot API exactly (no prefixes).

Methods that operate on the SceneTree itself (scene management, scene transitions) are categorized separately from Node/Object methods which operate on individual nodes.

## SceneTree Methods
Methods that manage the scene tree, scenes, and global state.

| Method | Returns | Description |
|--------|---------|-------------|
| `change_scene_to_file` | Error | Change to a scene file |
| `change_scene_to_node` | Error | Change to a packed scene node |
| `change_scene_to_packed` | Error | Change to a packed scene |
| `get_current_scene` | Node | Get the currently active scene |
| `get_edited_scene_root` | Node | Get the edited scene root node |
| `get_root` | Window | Get the root Window of the scene tree |
| `reload_current_scene` | Error | Reload the current scene |
| `set_current_scene` | Nil | Set the current scene |
| `set_edited_scene_root` | Nil | Set the edited scene root |
| `unload_current_scene` | Nil | Unload the current scene |

## Node Methods
Methods that operate on Node and Object instances.

| Method | Returns | Description |
|--------|---------|-------------|
| `add_user_signal` | Nil | Add a custom user signal |
| `call` | Nil | Call a method by name |
| `call_deferred` | Nil | Call a method on the next frame |
| `call_group` | Nil | Call a method on all nodes in a group |
| `call_group_flags` | Nil | Call a method on all nodes in a group with flags |
| `callv` | Nil | Call a method with an Array of arguments |
| `can_translate_messages` | bool | Check if messages can be translated |
| `cancel_free` | Nil | Cancel the freeing of a node |
| `connect` | Error | Connect a signal to a method |
| `create_timer` | SceneTreeTimer | Create a timer node |
| `create_tween` | Tween | Create a tween |
| `disconnect` | Nil | Disconnect a signal |
| `emit_signal` | Error | Emit a signal |
| `free` | Nil | Free the node |
| `get` | Nil | Get a property value |
| `get_class` | String | Get the class name |
| `get_first_node_in_group` | Node | Get the first node in a group |
| `get_frame` | int | Get the current frame |
| `get_incoming_connections` | Array | Get incoming signal connections |
| `get_indexed` | Nil | Get a child by index |
| `get_instance_id` | int | Get the instance ID |
| `get_meta` | Nil | Get a metadata value |
| `get_meta_list` | Array | Get all metadata keys |
| `get_method_argument_count` | int | Get the argument count of a method |
| `get_method_list` | Array | Get the list of methods |
| `get_multiplayer` | MultiplayerAPI | Get the multiplayer API |
| `get_node_count` | int | Get the number of child nodes |
| `get_node_count_in_group` | int | Get the number of nodes in a group |
| `get_nodes_in_group` | Array | Get all nodes in a group |
| `get_processed_tweens` | Array | Get all processed tweens |
| `get_property_list` | Array | Get the list of properties |
| `get_script` | Nil | Get the script attached to the node |
| `get_signal_connection_list` | Array | Get the list of signal connections |
| `get_signal_list` | Array | Get the list of signals |
| `get_translation_domain` | StringName | Get the translation domain |
| `has_connections` | bool | Check if there are connections |
| `has_group` | bool | Check if the node is in a group |
| `has_meta` | bool | Check if a metadata key exists |
| `has_method` | bool | Check if a method exists |
| `has_signal` | bool | Check if a signal exists |
| `has_user_signal` | bool | Check if a user signal exists |
| `is_accessibility_enabled` | bool | Check if accessibility is enabled |
| `is_accessibility_supported` | bool | Check if accessibility is supported |
| `is_auto_accept_quit` | bool | Check if auto-accept quit is enabled |
| `is_blocking_signals` | bool | Check if signals are blocked |
| `is_class` | bool | Check if the node is of a specific class |
| `is_connected` | bool | Check if a signal is connected |
| `is_debugging_collisions_hint` | bool | Check if collision debugging is enabled |
| `is_debugging_navigation_hint` | bool | Check if navigation debugging is enabled |
| `is_debugging_paths_hint` | bool | Check if path debugging is enabled |
| `is_multiplayer_poll_enabled` | bool | Check if multiplayer polling is enabled |
| `is_paused` | bool | Check if the scene is paused |
| `is_physics_interpolation_enabled` | bool | Check if physics interpolation is enabled |
| `is_queued_for_deletion` | bool | Check if the node is queued for deletion |
| `is_quit_on_go_back` | bool | Check if quit on go back is enabled |
| `notification` | Nil | Send a notification to the node |
| `notify_group` | Nil | Notify all nodes in a group |
| `notify_group_flags` | Nil | Notify all nodes in a group with flags |
| `notify_property_list_changed` | Nil | Notify that the property list changed |
| `property_can_revert` | bool | Check if a property can revert |
| `property_get_revert` | Nil | Get the revert value of a property |
| `queue_delete` | Nil | Queue the node for deletion |
| `quit` | Nil | Quit the application |
| `remove_meta` | Nil | Remove a metadata key |
| `remove_user_signal` | Nil | Remove a user signal |
| `set` | Nil | Set a property value |
| `set_auto_accept_quit` | Nil | Set auto-accept quit |
| `set_block_signals` | Nil | Set whether signals are blocked |
| `set_debug_collisions_hint` | Nil | Set collision debugging hint |
| `set_debug_navigation_hint` | Nil | Set navigation debugging hint |
| `set_debug_paths_hint` | Nil | Set path debugging hint |
| `set_deferred` | Nil | Set a property on the next frame |
| `set_group` | Nil | Set a group for the node |
| `set_group_flags` | Nil | Set group flags for the node |
| `set_indexed` | Nil | Set a child by index |
| `set_message_translation` | Nil | Set message translation |
| `set_meta` | Nil | Set a metadata value |
| `set_multiplayer` | Nil | Set the multiplayer API |
| `set_multiplayer_poll_enabled` | Nil | Set multiplayer polling enabled |
| `set_pause` | Nil | Set whether the scene is paused |
| `set_physics_interpolation_enabled` | Nil | Set physics interpolation enabled |
| `set_quit_on_go_back` | Nil | Set quit on go back |
| `set_script` | Nil | Set the script for the node |
| `set_translation_domain` | Nil | Set the translation domain |
| `to_string` | String | Convert to a string |
| `tr` | String | Translate a string |
| `tr_n` | String | Translate a string with context |
| `unload_current_scene` | Nil | Unload the current scene |

## Complete Method List
All available methods in alphabetical order.

| Method | Type |
|--------|------|
| `_finalize` | Virtual |
| `_get` | Virtual |
| `_get_property_list` | Virtual |
| `_init` | Virtual |
| `_initialize` | Virtual |
| `_iter_get` | Virtual |
| `_iter_init` | Virtual |
| `_iter_next` | Virtual |
| `_notification` | Virtual |
| `_physics_process` | Virtual |
| `_process` | Virtual |
| `_property_can_revert` | Virtual |
| `_property_get_revert` | Virtual |
| `_set` | Virtual |
| `_to_string` | Virtual |
| `_validate_property` | Virtual |
| `add_user_signal` | Node |
| `call` | Node |
| `call_deferred` | Node |
| `call_group` | Node |
| `call_group_flags` | Node |
| `callv` | Node |
| `can_translate_messages` | Node |
| `cancel_free` | Node |
| `change_scene_to_file` | SceneTree |
| `change_scene_to_node` | Node |
| `change_scene_to_packed` | Node |
| `connect` | Node |
| `create_timer` | Node |
| `create_tween` | Node |
| `disconnect` | Node |
| `emit_signal` | Node |
| `free` | Node |
| `get` | Node |
| `get_class` | Node |
| `get_current_scene` | SceneTree |
| `get_edited_scene_root` | SceneTree |
| `get_first_node_in_group` | Node |
| `get_frame` | Node |
| `get_incoming_connections` | Node |
| `get_indexed` | Node |
| `get_instance_id` | Node |
| `get_meta` | Node |
| `get_meta_list` | Node |
| `get_method_argument_count` | Node |
| `get_method_list` | Node |
| `get_multiplayer` | Node |
| `get_node_count` | Node |
| `get_node_count_in_group` | Node |
| `get_nodes_in_group` | Node |
| `get_processed_tweens` | Node |
| `get_property_list` | Node |
| `get_root` | SceneTree |
| `get_script` | Node |
| `get_signal_connection_list` | Node |
| `get_signal_list` | Node |
| `get_translation_domain` | Node |
| `has_connections` | Node |
| `has_group` | Node |
| `has_meta` | Node |
| `has_method` | Node |
| `has_signal` | Node |
| `has_user_signal` | Node |
| `is_accessibility_enabled` | Node |
| `is_accessibility_supported` | Node |
| `is_auto_accept_quit` | Node |
| `is_blocking_signals` | Node |
| `is_class` | Node |
| `is_connected` | Node |
| `is_debugging_collisions_hint` | Node |
| `is_debugging_navigation_hint` | Node |
| `is_debugging_paths_hint` | Node |
| `is_multiplayer_poll_enabled` | Node |
| `is_paused` | Node |
| `is_physics_interpolation_enabled` | Node |
| `is_queued_for_deletion` | Node |
| `is_quit_on_go_back` | Node |
| `notification` | Node |
| `notify_group` | Node |
| `notify_group_flags` | Node |
| `notify_property_list_changed` | Node |
| `property_can_revert` | Node |
| `property_get_revert` | Node |
| `queue_delete` | Node |
| `quit` | Node |
| `reload_current_scene` | SceneTree |
| `remove_meta` | Node |
| `remove_user_signal` | Node |
| `set` | Node |
| `set_auto_accept_quit` | Node |
| `set_block_signals` | Node |
| `set_current_scene` | SceneTree |
| `set_debug_collisions_hint` | Node |
| `set_debug_navigation_hint` | Node |
| `set_debug_paths_hint` | Node |
| `set_deferred` | Node |
| `set_edited_scene_root` | SceneTree |
| `set_group` | Node |
| `set_group_flags` | Node |
| `set_indexed` | Node |
| `set_message_translation` | Node |
| `set_meta` | Node |
| `set_multiplayer` | Node |
| `set_multiplayer_poll_enabled` | Node |
| `set_pause` | Node |
| `set_physics_interpolation_enabled` | Node |
| `set_quit_on_go_back` | Node |
| `set_script` | Node |
| `set_translation_domain` | Node |
| `to_string` | Node |
| `tr` | Node |
| `tr_n` | Node |
| `unload_current_scene` | SceneTree |

# SceneTree API Reference

## Overview

This document lists all 113 SceneTree and Object methods
that are exposed via the JSON-RPC interface.

## Method Index

- [_finalize](#_finalize) - `Nil _finalize()`
- [_get](#_get) - `Nil _get(StringName)`
- [_get_property_list](#_get_property_list) - `Array _get_property_list()`
- [_init](#_init) - `Nil _init()`
- [_initialize](#_initialize) - `Nil _initialize()`
- [_iter_get](#_iter_get) - `Nil _iter_get(Nil)`
- [_iter_init](#_iter_init) - `bool _iter_init(Array)`
- [_iter_next](#_iter_next) - `bool _iter_next(Array)`
- [_notification](#_notification) - `Nil _notification(int)`
- [_physics_process](#_physics_process) - `bool _physics_process(float)`
- [_process](#_process) - `bool _process(float)`
- [_property_can_revert](#_property_can_revert) - `bool _property_can_revert(StringName)`
- [_property_get_revert](#_property_get_revert) - `Nil _property_get_revert(StringName)`
- [_set](#_set) - `bool _set(StringName, Nil)`
- [_to_string](#_to_string) - `String _to_string()`
- [_validate_property](#_validate_property) - `Nil _validate_property(Dictionary)`
- [add_user_signal](#add_user_signal) - `Nil add_user_signal(String, Array)`
- [call](#call) - `Nil call(StringName)`
- [call_deferred](#call_deferred) - `Nil call_deferred(StringName)`
- [call_group](#call_group) - `Nil call_group(StringName, StringName)`
- [call_group_flags](#call_group_flags) - `Nil call_group_flags(int, StringName, StringName)`
- [callv](#callv) - `Nil callv(StringName, Array)`
- [can_translate_messages](#can_translate_messages) - `bool can_translate_messages()`
- [cancel_free](#cancel_free) - `Nil cancel_free()`
- [change_scene_to_file](#change_scene_to_file) - `int change_scene_to_file(String)`
- [change_scene_to_node](#change_scene_to_node) - `int change_scene_to_node(Object)`
- [change_scene_to_packed](#change_scene_to_packed) - `int change_scene_to_packed(Object)`
- [connect](#connect) - `int connect(StringName, Callable, int)`
- [create_timer](#create_timer) - `Object create_timer(float, bool, bool, bool)`
- [create_tween](#create_tween) - `Object create_tween()`
- [disconnect](#disconnect) - `Nil disconnect(StringName, Callable)`
- [emit_signal](#emit_signal) - `int emit_signal(StringName)`
- [free](#free) - `Nil free()`
- [get](#get) - `Nil get(StringName)`
- [get_class](#get_class) - `String get_class()`
- [get_current_scene](#get_current_scene) - `Object get_current_scene()`
- [get_edited_scene_root](#get_edited_scene_root) - `Object get_edited_scene_root()`
- [get_first_node_in_group](#get_first_node_in_group) - `Object get_first_node_in_group(StringName)`
- [get_frame](#get_frame) - `int get_frame()`
- [get_incoming_connections](#get_incoming_connections) - `Array get_incoming_connections()`
- [get_indexed](#get_indexed) - `Nil get_indexed(NodePath)`
- [get_instance_id](#get_instance_id) - `int get_instance_id()`
- [get_meta](#get_meta) - `Nil get_meta(StringName, Nil)`
- [get_meta_list](#get_meta_list) - `Array get_meta_list()`
- [get_method_argument_count](#get_method_argument_count) - `int get_method_argument_count(StringName)`
- [get_method_list](#get_method_list) - `Array get_method_list()`
- [get_multiplayer](#get_multiplayer) - `Object get_multiplayer(NodePath)`
- [get_node_count](#get_node_count) - `int get_node_count()`
- [get_node_count_in_group](#get_node_count_in_group) - `int get_node_count_in_group(StringName)`
- [get_nodes_in_group](#get_nodes_in_group) - `Array get_nodes_in_group(StringName)`
- [get_processed_tweens](#get_processed_tweens) - `Array get_processed_tweens()`
- [get_property_list](#get_property_list) - `Array get_property_list()`
- [get_root](#get_root) - `Object get_root()`
- [get_script](#get_script) - `Nil get_script()`
- [get_signal_connection_list](#get_signal_connection_list) - `Array get_signal_connection_list(StringName)`
- [get_signal_list](#get_signal_list) - `Array get_signal_list()`
- [get_translation_domain](#get_translation_domain) - `StringName get_translation_domain()`
- [has_connections](#has_connections) - `bool has_connections(StringName)`
- [has_group](#has_group) - `bool has_group(StringName)`
- [has_meta](#has_meta) - `bool has_meta(StringName)`
- [has_method](#has_method) - `bool has_method(StringName)`
- [has_signal](#has_signal) - `bool has_signal(StringName)`
- [has_user_signal](#has_user_signal) - `bool has_user_signal(StringName)`
- [is_accessibility_enabled](#is_accessibility_enabled) - `bool is_accessibility_enabled()`
- [is_accessibility_supported](#is_accessibility_supported) - `bool is_accessibility_supported()`
- [is_auto_accept_quit](#is_auto_accept_quit) - `bool is_auto_accept_quit()`
- [is_blocking_signals](#is_blocking_signals) - `bool is_blocking_signals()`
- [is_class](#is_class) - `bool is_class(String)`
- [is_connected](#is_connected) - `bool is_connected(StringName, Callable)`
- [is_debugging_collisions_hint](#is_debugging_collisions_hint) - `bool is_debugging_collisions_hint()`
- [is_debugging_navigation_hint](#is_debugging_navigation_hint) - `bool is_debugging_navigation_hint()`
- [is_debugging_paths_hint](#is_debugging_paths_hint) - `bool is_debugging_paths_hint()`
- [is_multiplayer_poll_enabled](#is_multiplayer_poll_enabled) - `bool is_multiplayer_poll_enabled()`
- [is_paused](#is_paused) - `bool is_paused()`
- [is_physics_interpolation_enabled](#is_physics_interpolation_enabled) - `bool is_physics_interpolation_enabled()`
- [is_queued_for_deletion](#is_queued_for_deletion) - `bool is_queued_for_deletion()`
- [is_quit_on_go_back](#is_quit_on_go_back) - `bool is_quit_on_go_back()`
- [notification](#notification) - `Nil notification(int, bool)`
- [notify_group](#notify_group) - `Nil notify_group(StringName, int)`
- [notify_group_flags](#notify_group_flags) - `Nil notify_group_flags(int, StringName, int)`
- [notify_property_list_changed](#notify_property_list_changed) - `Nil notify_property_list_changed()`
- [property_can_revert](#property_can_revert) - `bool property_can_revert(StringName)`
- [property_get_revert](#property_get_revert) - `Nil property_get_revert(StringName)`
- [queue_delete](#queue_delete) - `Nil queue_delete(Object)`
- [quit](#quit) - `Nil quit(int)`
- [reload_current_scene](#reload_current_scene) - `int reload_current_scene()`
- [remove_meta](#remove_meta) - `Nil remove_meta(StringName)`
- [remove_user_signal](#remove_user_signal) - `Nil remove_user_signal(StringName)`
- [set](#set) - `Nil set(StringName, Nil)`
- [set_auto_accept_quit](#set_auto_accept_quit) - `Nil set_auto_accept_quit(bool)`
- [set_block_signals](#set_block_signals) - `Nil set_block_signals(bool)`
- [set_current_scene](#set_current_scene) - `Nil set_current_scene(Object)`
- [set_debug_collisions_hint](#set_debug_collisions_hint) - `Nil set_debug_collisions_hint(bool)`
- [set_debug_navigation_hint](#set_debug_navigation_hint) - `Nil set_debug_navigation_hint(bool)`
- [set_debug_paths_hint](#set_debug_paths_hint) - `Nil set_debug_paths_hint(bool)`
- [set_deferred](#set_deferred) - `Nil set_deferred(StringName, Nil)`
- [set_edited_scene_root](#set_edited_scene_root) - `Nil set_edited_scene_root(Object)`
- [set_group](#set_group) - `Nil set_group(StringName, String, Nil)`
- [set_group_flags](#set_group_flags) - `Nil set_group_flags(int, StringName, String, Nil)`
- [set_indexed](#set_indexed) - `Nil set_indexed(NodePath, Nil)`
- [set_message_translation](#set_message_translation) - `Nil set_message_translation(bool)`
- [set_meta](#set_meta) - `Nil set_meta(StringName, Nil)`
- [set_multiplayer](#set_multiplayer) - `Nil set_multiplayer(Object, NodePath)`
- [set_multiplayer_poll_enabled](#set_multiplayer_poll_enabled) - `Nil set_multiplayer_poll_enabled(bool)`
- [set_pause](#set_pause) - `Nil set_pause(bool)`
- [set_physics_interpolation_enabled](#set_physics_interpolation_enabled) - `Nil set_physics_interpolation_enabled(bool)`
- [set_quit_on_go_back](#set_quit_on_go_back) - `Nil set_quit_on_go_back(bool)`
- [set_script](#set_script) - `Nil set_script(Nil)`
- [set_translation_domain](#set_translation_domain) - `Nil set_translation_domain(StringName)`
- [to_string](#to_string) - `String to_string()`
- [tr](#tr) - `String tr(StringName, StringName)`
- [tr_n](#tr_n) - `String tr_n(StringName, StringName, int, StringName)`
- [unload_current_scene](#unload_current_scene) - `Nil unload_current_scene()`

---

## _finalize

**Method ID:** 0
**Flags:** VIRTUAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _get

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `StringName property`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _get_property_list

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## _init

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _initialize

**Method ID:** 0
**Flags:** VIRTUAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _iter_get

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `Nil iter`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _iter_init

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `Array iter`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _iter_next

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `Array iter`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _notification

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `int what`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _physics_process

**Method ID:** 0
**Flags:** VIRTUAL

### Arguments

- `float delta`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _process

**Method ID:** 0
**Flags:** VIRTUAL

### Arguments

- `float delta`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _property_can_revert

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `StringName property`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _property_get_revert

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `StringName property`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## _set

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `StringName property`
- `Nil value`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## _to_string

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Returns

`String`

*Method names match Godot 4.6.1 API exactly*

---

## _validate_property

**Method ID:** 0
**Flags:** NORMAL, VIRTUAL, OBJECT_CORE

### Arguments

- `Dictionary property`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## add_user_signal

**Method ID:** 20
**Flags:** NORMAL

### Arguments

- `String signal`
- `Array arguments`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## call

**Method ID:** 24
**Flags:** NORMAL, VARARG

### Arguments

- `StringName method`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## call_deferred

**Method ID:** 25
**Flags:** NORMAL, VARARG

### Arguments

- `StringName method`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## call_group

**Method ID:** 13667
**Flags:** NORMAL, VARARG

### Arguments

- `StringName group`
- `StringName method`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## call_group_flags

**Method ID:** 13664
**Flags:** NORMAL, VARARG

### Arguments

- `int flags`
- `StringName group`
- `StringName method`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## callv

**Method ID:** 27
**Flags:** NORMAL

### Arguments

- `StringName method`
- `Array arg_array`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## can_translate_messages

**Method ID:** 42
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## cancel_free

**Method ID:** 48
**Flags:** NORMAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## change_scene_to_file

**Method ID:** 13675
**Flags:** NORMAL

### Arguments

- `String path`

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## change_scene_to_node

**Method ID:** 13677
**Flags:** NORMAL

### Arguments

- `Object node`

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## change_scene_to_packed

**Method ID:** 13676
**Flags:** NORMAL

### Arguments

- `Object packed_scene`

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## connect

**Method ID:** 34
**Flags:** NORMAL

### Arguments

- `StringName signal`
- `Callable callable`
- `int flags`

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## create_timer

**Method ID:** 13655
**Flags:** NORMAL

### Arguments

- `float time_sec`
- `bool process_always`
- `bool process_in_physics`
- `bool ignore_time_scale`

### Returns

`Object`
- Class: `SceneTreeTimer`

*Method names match Godot 4.6.1 API exactly*

---

## create_tween

**Method ID:** 13656
**Flags:** NORMAL

### Returns

`Object`
- Class: `Tween`

*Method names match Godot 4.6.1 API exactly*

---

## disconnect

**Method ID:** 35
**Flags:** NORMAL

### Arguments

- `StringName signal`
- `Callable callable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## emit_signal

**Method ID:** 23
**Flags:** NORMAL, VARARG

### Arguments

- `StringName signal`

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## free

**Method ID:** 0
**Flags:** NORMAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## get

**Method ID:** 3
**Flags:** NORMAL, CONST

### Arguments

- `StringName property`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## get_class

**Method ID:** 0
**Flags:** NORMAL, CONST

### Returns

`String`

*Method names match Godot 4.6.1 API exactly*

---

## get_current_scene

**Method ID:** 13674
**Flags:** NORMAL, CONST

### Returns

`Object`
- Class: `Node`

*Method names match Godot 4.6.1 API exactly*

---

## get_edited_scene_root

**Method ID:** 13652
**Flags:** NORMAL, CONST

### Returns

`Object`
- Class: `Node`

*Method names match Godot 4.6.1 API exactly*

---

## get_first_node_in_group

**Method ID:** 13671
**Flags:** NORMAL

### Arguments

- `StringName group`

### Returns

`Object`
- Class: `Node`

*Method names match Godot 4.6.1 API exactly*

---

## get_frame

**Method ID:** 13659
**Flags:** NORMAL, CONST

### Returns

`int`

*Method names match Godot 4.6.1 API exactly*

---

## get_incoming_connections

**Method ID:** 33
**Flags:** NORMAL, CONST

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_indexed

**Method ID:** 5
**Flags:** NORMAL, CONST

### Arguments

- `NodePath property_path`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## get_instance_id

**Method ID:** 12
**Flags:** NORMAL, CONST

### Returns

`int`

*Method names match Godot 4.6.1 API exactly*

---

## get_meta

**Method ID:** 17
**Flags:** NORMAL, CONST

### Arguments

- `StringName name`
- `Nil default`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## get_meta_list

**Method ID:** 19
**Flags:** NORMAL, CONST

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_method_argument_count

**Method ID:** 29
**Flags:** NORMAL, CONST

### Arguments

- `StringName method`

### Returns

`int`

*Method names match Godot 4.6.1 API exactly*

---

## get_method_list

**Method ID:** 7
**Flags:** NORMAL, CONST

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_multiplayer

**Method ID:** 13681
**Flags:** NORMAL, CONST

### Arguments

- `NodePath for_path`

### Returns

`Object`
- Class: `MultiplayerAPI`

*Method names match Godot 4.6.1 API exactly*

---

## get_node_count

**Method ID:** 13658
**Flags:** NORMAL, CONST

### Returns

`int`

*Method names match Godot 4.6.1 API exactly*

---

## get_node_count_in_group

**Method ID:** 13672
**Flags:** NORMAL, CONST

### Arguments

- `StringName group`

### Returns

`int`

*Method names match Godot 4.6.1 API exactly*

---

## get_nodes_in_group

**Method ID:** 13670
**Flags:** NORMAL

### Arguments

- `StringName group`

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_processed_tweens

**Method ID:** 13657
**Flags:** NORMAL

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_property_list

**Method ID:** 6
**Flags:** NORMAL, CONST

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_root

**Method ID:** 13637
**Flags:** NORMAL, CONST

### Returns

`Object`
- Class: `Window`

*Method names match Godot 4.6.1 API exactly*

---

## get_script

**Method ID:** 14
**Flags:** NORMAL, CONST

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## get_signal_connection_list

**Method ID:** 32
**Flags:** NORMAL, CONST

### Arguments

- `StringName signal`

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_signal_list

**Method ID:** 31
**Flags:** NORMAL, CONST

### Returns

`Array`

*Method names match Godot 4.6.1 API exactly*

---

## get_translation_domain

**Method ID:** 45
**Flags:** NORMAL, CONST

### Returns

`StringName`

*Method names match Godot 4.6.1 API exactly*

---

## has_connections

**Method ID:** 37
**Flags:** NORMAL, CONST

### Arguments

- `StringName signal`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## has_group

**Method ID:** 13638
**Flags:** NORMAL, CONST

### Arguments

- `StringName name`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## has_meta

**Method ID:** 18
**Flags:** NORMAL, CONST

### Arguments

- `StringName name`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## has_method

**Method ID:** 28
**Flags:** NORMAL, CONST

### Arguments

- `StringName method`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## has_signal

**Method ID:** 30
**Flags:** NORMAL, CONST

### Arguments

- `StringName signal`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## has_user_signal

**Method ID:** 21
**Flags:** NORMAL, CONST

### Arguments

- `StringName signal`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_accessibility_enabled

**Method ID:** 13639
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_accessibility_supported

**Method ID:** 13640
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_auto_accept_quit

**Method ID:** 13641
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_blocking_signals

**Method ID:** 39
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_class

**Method ID:** 1
**Flags:** NORMAL, CONST

### Arguments

- `String class`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_connected

**Method ID:** 36
**Flags:** NORMAL, CONST

### Arguments

- `StringName signal`
- `Callable callable`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_debugging_collisions_hint

**Method ID:** 13646
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_debugging_navigation_hint

**Method ID:** 13650
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_debugging_paths_hint

**Method ID:** 13648
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_multiplayer_poll_enabled

**Method ID:** 13683
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_paused

**Method ID:** 13654
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_physics_interpolation_enabled

**Method ID:** 13662
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_queued_for_deletion

**Method ID:** 47
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## is_quit_on_go_back

**Method ID:** 13643
**Flags:** NORMAL, CONST

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## notification

**Method ID:** 10
**Flags:** NORMAL

### Arguments

- `int what`
- `bool reversed`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## notify_group

**Method ID:** 13668
**Flags:** NORMAL

### Arguments

- `StringName group`
- `int notification`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## notify_group_flags

**Method ID:** 13665
**Flags:** NORMAL

### Arguments

- `int call_flags`
- `StringName group`
- `int notification`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## notify_property_list_changed

**Method ID:** 40
**Flags:** NORMAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## property_can_revert

**Method ID:** 8
**Flags:** NORMAL, CONST

### Arguments

- `StringName property`

### Returns

`bool`

*Method names match Godot 4.6.1 API exactly*

---

## property_get_revert

**Method ID:** 9
**Flags:** NORMAL, CONST

### Arguments

- `StringName property`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## queue_delete

**Method ID:** 13663
**Flags:** NORMAL

### Arguments

- `Object obj`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## quit

**Method ID:** 13660
**Flags:** NORMAL

### Arguments

- `int exit_code`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## reload_current_scene

**Method ID:** 13678
**Flags:** NORMAL

### Returns

`int`
- Class: `Error`

*Method names match Godot 4.6.1 API exactly*

---

## remove_meta

**Method ID:** 16
**Flags:** NORMAL

### Arguments

- `StringName name`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## remove_user_signal

**Method ID:** 22
**Flags:** NORMAL

### Arguments

- `StringName signal`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set

**Method ID:** 2
**Flags:** NORMAL

### Arguments

- `StringName property`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_auto_accept_quit

**Method ID:** 13642
**Flags:** NORMAL

### Arguments

- `bool enabled`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_block_signals

**Method ID:** 38
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_current_scene

**Method ID:** 13673
**Flags:** NORMAL

### Arguments

- `Object child_node`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_debug_collisions_hint

**Method ID:** 13645
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_debug_navigation_hint

**Method ID:** 13649
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_debug_paths_hint

**Method ID:** 13647
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_deferred

**Method ID:** 26
**Flags:** NORMAL

### Arguments

- `StringName property`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_edited_scene_root

**Method ID:** 13651
**Flags:** NORMAL

### Arguments

- `Object scene`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_group

**Method ID:** 13669
**Flags:** NORMAL

### Arguments

- `StringName group`
- `String property`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_group_flags

**Method ID:** 13666
**Flags:** NORMAL

### Arguments

- `int call_flags`
- `StringName group`
- `String property`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_indexed

**Method ID:** 4
**Flags:** NORMAL

### Arguments

- `NodePath property_path`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_message_translation

**Method ID:** 41
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_meta

**Method ID:** 15
**Flags:** NORMAL

### Arguments

- `StringName name`
- `Nil value`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_multiplayer

**Method ID:** 13680
**Flags:** NORMAL

### Arguments

- `Object multiplayer`
- `NodePath root_path`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_multiplayer_poll_enabled

**Method ID:** 13682
**Flags:** NORMAL

### Arguments

- `bool enabled`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_pause

**Method ID:** 13653
**Flags:** NORMAL

### Arguments

- `bool enable`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_physics_interpolation_enabled

**Method ID:** 13661
**Flags:** NORMAL

### Arguments

- `bool enabled`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_quit_on_go_back

**Method ID:** 13644
**Flags:** NORMAL

### Arguments

- `bool enabled`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_script

**Method ID:** 13
**Flags:** NORMAL

### Arguments

- `Nil script`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## set_translation_domain

**Method ID:** 46
**Flags:** NORMAL

### Arguments

- `StringName domain`

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

## to_string

**Method ID:** 11
**Flags:** NORMAL

### Returns

`String`

*Method names match Godot 4.6.1 API exactly*

---

## tr

**Method ID:** 43
**Flags:** NORMAL, CONST

### Arguments

- `StringName message`
- `StringName context`

### Returns

`String`

*Method names match Godot 4.6.1 API exactly*

---

## tr_n

**Method ID:** 44
**Flags:** NORMAL, CONST

### Arguments

- `StringName message`
- `StringName plural_message`
- `int n`
- `StringName context`

### Returns

`String`

*Method names match Godot 4.6.1 API exactly*

---

## unload_current_scene

**Method ID:** 13679
**Flags:** NORMAL

### Returns

`Nil`

*Method names match Godot 4.6.1 API exactly*

---

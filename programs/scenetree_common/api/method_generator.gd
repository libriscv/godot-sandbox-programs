extends Node

# Dictionary mapping for MethodFlags
const METHOD_FLAGS_MAP: Dictionary = {
	METHOD_FLAG_NORMAL: "METHOD_FLAG_NORMAL",
	METHOD_FLAG_EDITOR: "METHOD_FLAG_EDITOR",
	METHOD_FLAG_CONST: "METHOD_FLAG_CONST",
	METHOD_FLAG_VIRTUAL: "METHOD_FLAG_VIRTUAL",
	METHOD_FLAG_VARARG: "METHOD_FLAG_VARARG",
	METHOD_FLAG_STATIC: "METHOD_FLAG_STATIC",
	METHOD_FLAG_OBJECT_CORE: "METHOD_FLAG_OBJECT_CORE"
}

# Dictionary mapping for common PropertyUsageFlags
const PROPERTY_USAGE_MAP: Dictionary = {
	PROPERTY_USAGE_STORAGE: "PROPERTY_USAGE_STORAGE",
	PROPERTY_USAGE_EDITOR: "PROPERTY_USAGE_EDITOR",
	PROPERTY_USAGE_INTERNAL: "PROPERTY_USAGE_INTERNAL",
	PROPERTY_USAGE_CHECKABLE: "PROPERTY_USAGE_CHECKABLE",
	PROPERTY_USAGE_CHECKED: "PROPERTY_USAGE_CHECKED",
	PROPERTY_USAGE_GROUP: "PROPERTY_USAGE_GROUP",
	PROPERTY_USAGE_CATEGORY: "PROPERTY_USAGE_CATEGORY",
	PROPERTY_USAGE_SUBGROUP: "PROPERTY_USAGE_SUBGROUP",
	PROPERTY_USAGE_CLASS_IS_BITFIELD: "PROPERTY_USAGE_CLASS_IS_BITFIELD",
	PROPERTY_USAGE_NO_INSTANCE_STATE: "PROPERTY_USAGE_NO_INSTANCE_STATE",
	PROPERTY_USAGE_RESTART_IF_CHANGED: "PROPERTY_USAGE_RESTART_IF_CHANGED",
	PROPERTY_USAGE_SCRIPT_VARIABLE: "PROPERTY_USAGE_SCRIPT_VARIABLE",
	PROPERTY_USAGE_READ_ONLY: "PROPERTY_USAGE_READ_ONLY"
}

func _ready() -> void:
	var scene_tree: SceneTree = get_tree()
	
	var public_file := FileAccess.open("res://public_methods_v4.6.1_stable.jsonl", FileAccess.WRITE)
	var virtual_file := FileAccess.open("res://virtual_methods_v4.6.1_stable.jsonl", FileAccess.WRITE)
	
	if not public_file or not virtual_file:
		push_error("Failed to open files for writing.")
		return

	for m: Dictionary in scene_tree.get_method_list():
		var method_data: Dictionary = m.duplicate(true)
		
		# 1. Decompose Method Flags
		if method_data.has("flags"):
			method_data["flags"] = _decompose_bitmask(method_data["flags"], METHOD_FLAGS_MAP)

		# 2. Process Return values (Type and Usage Bitmask)
		if method_data.has("return"):
			if method_data["return"].has("type"):
				method_data["return"]["type"] = type_string(method_data["return"]["type"])
			if method_data["return"].has("usage"):
				method_data["return"]["usage"] = _decompose_bitmask(method_data["return"]["usage"], PROPERTY_USAGE_MAP)
			
		# 3. Process Arguments (Type and Usage Bitmask)
		if method_data.has("args"):
			for arg in method_data["args"]:
				if arg.has("type"):
					arg["type"] = type_string(arg["type"])
				if arg.has("usage"):
					arg["usage"] = _decompose_bitmask(arg["usage"], PROPERTY_USAGE_MAP)

		var json_string: String = JSON.stringify(method_data)
		
		# 4. Sort and Filter
		var is_virtual: bool = "METHOD_FLAG_VIRTUAL" in method_data["flags"]
		
		if is_virtual:
			virtual_file.store_line(json_string)
		elif String(method_data.name).begins_with("_"):
			# Filters out non-virtual internal/private methods entirely
			pass
		else:
			public_file.store_line(json_string)
			
	public_file.close()
	virtual_file.close()
	
	print("Methods exported with decomposed bitmasks!")


### Generic Bitmask Decoder
func _decompose_bitmask(mask: int, flag_map: Dictionary) -> Array[String]:
	var active_flags: Array[String] = []
	
	# Handle the special case where the mask is completely empty (0)
	if mask == 0:
		return ["NONE"]
		
	for flag_value: int in flag_map.keys():
		# Use bitwise AND to check if the specific flag is "on" inside the mask
		if (mask & flag_value) == flag_value and flag_value != 0:
			active_flags.append(flag_map[flag_value])
			
	return active_flags

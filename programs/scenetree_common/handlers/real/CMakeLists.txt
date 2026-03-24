# ==================== Real SceneTree Handlers ====================
# Real implementations that call into Godot's C++ API.
# These are used by the embedded godot-sandbox mode.

cmake_minimum_required(VERSION 3.15)
project(handlers_real)

# These handlers are implemented in the main scenetree_passthrough
# They just forward to the existing handlers.cpp implementation
# For now, we just create an empty library that links to the main program

# This directory intentionally has minimal code since most handlers
# are implemented in the main scenetree_passthrough program.

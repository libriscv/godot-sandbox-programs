#pragma once

// ==================== Base Handler Class ====================
// Provides common handler infrastructure for SceneTree passthrough.
// Both the Godot-integrated and standalone versions inherit from this.

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

// Buffer size for building JSON responses
#define JSONRPC_BUFFER_SIZE 65536

// Opaque handles for Godot types (avoids including godot headers)
typedef struct _GodotObject GodotObject;
typedef struct _GodotNode GodotNode;
typedef struct _GodotSceneTree GodotSceneTree;
typedef struct _GodotVariant GodotVariant;

// ==================== Result Type ====================

typedef struct {
    bool is_error;
    int error_code;
    const char* error_message;
    const char* result_json;  // JSON string of the result
} HandlerResult;

// ==================== Error Constructors ====================

static HandlerResult make_handler_error(int code, const char* message) {
    HandlerResult r;
    r.is_error = true;
    r.error_code = code;
    r.error_message = message;
    r.result_json = NULL;
    return r;
}

static HandlerResult make_handler_ok(const char* json_result) {
    HandlerResult r;
    r.is_error = false;
    r.error_code = 0;
    r.error_message = NULL;
    r.result_json = json_result;
    return r;
}

// ==================== JSON Building Helpers ====================
// All static inline for efficiency

static inline void json_array_start(char* buf, size_t* len) {
    buf[(*len)++] = '[';
}

static inline void json_array_end(char* buf, size_t* len) {
    buf[(*len)++] = ']';
    buf[*len] = '\0';
}

static inline void json_object_start(char* buf, size_t* len) {
    buf[(*len)++] = '{';
}

static inline void json_object_end(char* buf, size_t* len) {
    buf[(*len)++] = '}';
    buf[*len] = '\0';
}

static inline void json_add_comma(char* buf, size_t* len) {
    if (*len > 1) {
        buf[(*len)++] = ',';
    }
}

static inline void json_add_key(char* buf, size_t* len, const char* key) {
    json_add_comma(buf, len);
    *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\"%s\":", key);
}

static inline void json_add_string(char* buf, size_t* len, const char* value) {
    json_add_comma(buf, len);
    
    buf[(*len)++] = '"';
    const char* p = value;
    while (p && *p) {
        switch (*p) {
            case '"':  *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\\""); break;
            case '\\': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\\\"); break;
            case '\b': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\b"); break;
            case '\f': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\f"); break;
            case '\n': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\n"); break;
            case '\r': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\r"); break;
            case '\t': *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "\\t"); break;
            default:   buf[(*len)++] = *p; break;
        }
        p++;
    }
    buf[(*len)++] = '"';
    buf[*len] = '\0';
}

static inline void json_add_int(char* buf, size_t* len, int64_t value) {
    json_add_comma(buf, len);
    *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "%lld", value);
}

static inline void json_add_bool(char* buf, size_t* len, bool value) {
    json_add_comma(buf, len);
    *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "%s", value ? "true" : "false");
}

static inline void json_add_null(char* buf, size_t* len) {
    json_add_comma(buf, len);
    *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "null");
}

static inline void json_add_double(char* buf, size_t* len, double value) {
    json_add_comma(buf, len);
    *len += snprintf(buf + *len, JSONRPC_BUFFER_SIZE - *len, "%f", value);
}

static inline void json_add_key_string(char* buf, size_t* len, const char* key, const char* value) {
    json_add_key(buf, len, key);
    json_add_string(buf, len, value ? value : "");
}

static inline void json_add_key_int(char* buf, size_t* len, const char* key, int64_t value) {
    json_add_key(buf, len, key);
    json_add_int(buf, len, value);
}

static inline void json_add_key_bool(char* buf, size_t* len, const char* key, bool value) {
    json_add_key(buf, len, key);
    json_add_bool(buf, len, value);
}

static inline void json_add_key_double(char* buf, size_t* len, const char* key, double value) {
    json_add_key(buf, len, key);
    json_add_double(buf, len, value);
}

// ==================== String Escaping ====================

static inline void escape_json_string(char* dst, const char* src, size_t dst_size) {
    if (!dst || !src || dst_size == 0) return;
    
    char* d = dst;
    const char* s = src;
    size_t remaining = dst_size - 1; // Leave room for null terminator
    
    *d++ = '"';
    remaining--;
    
    while (*s && remaining > 1) {
        switch (*s) {
            case '"':  *d++ = '\\'; *d++ = '"'; remaining -= 2; break;
            case '\\': *d++ = '\\'; *d++ = '\\'; remaining -= 2; break;
            case '\b': *d++ = '\\'; *d++ = 'b'; remaining -= 2; break;
            case '\f': *d++ = '\\'; *d++ = 'f'; remaining -= 2; break;
            case '\n': *d++ = '\\'; *d++ = 'n'; remaining -= 2; break;
            case '\r': *d++ = '\\'; *d++ = 'r'; remaining -= 2; break;
            case '\t': *d++ = '\\'; *d++ = 't'; remaining -= 2; break;
            default:   *d++ = *s; remaining--; break;
        }
        s++;
    }
    
    *d++ = '"';
    *d = '\0';
}

// ==================== Variant Type Helpers ====================

static inline const char* variant_type_to_string(int type) {
    switch (type) {
        case 0:  return "Nil";
        case 1:  return "Bool";
        case 2:  return "int";
        case 3:  return "float";
        case 4:  return "String";
        case 5:  return "Vector2";
        case 6:  return "Vector3";
        case 7:  return "Color";
        case 8:  return "NodePath";
        case 9:  return "Object";
        case 10: return "Dictionary";
        case 11: return "Array";
        default: return "Unknown";
    }
}

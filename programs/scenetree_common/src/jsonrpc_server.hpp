#pragma once

// ==================== JSON-RPC 2.0 Server Framework ====================
// Common JSON-RPC infrastructure shared between standalone and embedded modes.
// This provides the dispatch loop, request parsing, and response formatting.
//
// Modes of operation:
// 1. TCP Server mode - listens on a socket, accepts JSON-RPC over TCP
// 2. Stdio mode - reads from stdin, writes to stdout (for pipes/fifos)
// 3. Embedded mode - dispatches to handlers via callv ABI (for Godot integration)

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <pthread.h>

// ==================== Constants ====================

#define JSONRPC_BUFFER_SIZE 65536
#define JSONRPC_MAX_ID_LENGTH 256

// Error codes (JSON-RPC 2.0 standard)
#define JSONRPC_PARSE_ERROR -32700
#define JSONRPC_INVALID_REQUEST -32600
#define JSONRPC_METHOD_NOT_FOUND -32601
#define JSONRPC_INVALID_PARAMS -32602
#define JSONRPC_INTERNAL_ERROR -32603

// ==================== Types ====================

typedef struct {
    const char* method;
    int id;
    char params_json[JSONRPC_BUFFER_SIZE];
} JSONRPCRequest;

// Forward declaration for handler type
typedef const char* (*JSONRPCHandlerFunc)(const char* method, const char* params_json, int id);

// ==================== Server State ====================

typedef struct {
    int sockfd;              // TCP socket file descriptor (0 if not TCP)
    FILE* logfile;           // Log file handle
    bool use_stdio;          // Use stdin/stdout instead of TCP
    int port;                // TCP port number
    JSONRPCHandlerFunc handler;  // Request handler function
} JSONRPCServerState;

// Global server state
extern JSONRPCServerState jsonrpc_server;

// ==================== API ====================

// Initialize the server state
void jsonrpc_server_init(int port, bool use_stdio, FILE* logfile, JSONRPCHandlerFunc handler);

// Start the server loop
// Returns when the server shuts down
void jsonrpc_server_run(void);

// Send a JSON-RPC response
void jsonrpc_send_response(int id, const char* result_json);

// Send a JSON-RPC error
void jsonrpc_send_error(int id, int code, const char* message, const char* data);

// Log a message
void jsonrpc_log(const char* format, ...);

// Parse a JSON-RPC request from text
// Returns 0 on success, -1 on parse error
int jsonrpc_parse_request(const char* request_text, JSONRPCRequest* out_request);

// Helper to extract string from JSON
const char* json_extract_string(const char* json, const char* key);

// Helper to extract int from JSON
int json_extract_int(const char* json, const char* key, int default_value);

// Helper to extract bool from JSON
bool json_extract_bool(const char* json, const char* key, bool default_value);

// ==================== Utility Macros ====================

// Define a handler function
// The handler should return a JSON string (the result value)
// Example: "42", "\"hello\"", "[1,2,3]"
#define JSONRPC_HANDLER(name) const char* handle_##name(const char* params_json, int id)

// Register a handler in the dispatch table
#define JSONRPC_REGISTER_HANDLER(map, name) \
    do { \
        (map)[#name] = &handle_##name; \
    } while (0)

// Dispatch a request (call the appropriate handler)
#define JSONRPC_DISPATCH(map, request) \
    do { \
        const char* handler_name = json_extract_string((request)->method, "method"); \
        JSONRPCHandlerFunc handler = (map)[handler_name]; \
        if (handler) { \
            const char* result = handler((request)->params_json, (request)->id); \
            jsonrpc_send_response((request)->id, result); \
        } else { \
            jsonrpc_send_error((request)->id, JSONRPC_METHOD_NOT_FOUND, \
                              "Method not found", handler_name); \
        } \
    } while (0)

// ==================== Standalone SceneTree Passthrough JSON-RPC Server ====================
// A standalone JSON-RPC 2.0 server that exposes Godot's SceneTree and Node API.
//
// Build: cmake -S . -B ../../.build_standalone && make -C ../../.build_standalone
// Run: ./../../.build_standalone/bin/scenetree_passthrough_standalone --stdio
//
// This version uses the common JSON-RPC server framework (scenetree_common) with
// mock handlers for testing without Godot API access.

#include "../../scenetree_common/src/jsonrpc_server.hpp"
#include "../../scenetree_common/include/handler_base.hpp"
#include "../../scenetree_common/handlers/mock/mock_handlers.cpp"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

// ==================== Configuration ====================

static int g_port = 7777;
static bool g_use_stdio = false;
static const char* g_logfile_path = NULL;

// ==================== Main ====================

static void print_usage(const char* argv0) {
    printf("Usage: %s [options]\n", argv0);
    printf("\n");
    printf("Standalone SceneTree Passthrough JSON-RPC Server\n");
    printf("\n");
    printf("Options:\n");
    printf("  -p, --port <port>    TCP port to listen on (default: 7777)\n");
    printf("  -s, --stdio          Use stdin/stdout mode (for pipes/fifos)\n");
    printf("  -l, --log <file>     Log to file\n");
    printf("  -h, --help           Show this help\n");
    printf("\n");
    printf("Demo commands (stdio mode):\n");
    printf("  {\"jsonrpc\":\"2.0\",\"method\":\"list_methods\",\"id\":1}\n");
    printf("  {\"jsonrpc\":\"2.0\",\"method\":\"get_tree\",\"id\":2}\n");
    printf("\n");
}

int main(int argc, char** argv) {
    // Parse arguments using switch-like pattern on argv values
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            print_usage(argv[0]);
            return 0;
        } else if ((strcmp(argv[i], "-s") == 0) || (strcmp(argv[i], "--stdio") == 0)) {
            g_use_stdio = true;
        } else if ((strcmp(argv[i], "-p") == 0) || (strcmp(argv[i], "--port") == 0)) {
            if ((i + 1) < argc) {
                g_port = atoi(argv[++i]);
            }
        } else if ((strcmp(argv[i], "-l") == 0) || (strcmp(argv[i], "--log") == 0)) {
            if ((i + 1) < argc) {
                g_logfile_path = argv[++i];
            }
        }
    }
    
    // Initialize server
    FILE* logfile = NULL;
    if (g_logfile_path != NULL) {
        logfile = fopen(g_logfile_path, "a");
        if (!(logfile != NULL)) {
            fprintf(stderr, "Warning: Could not open log file '%s': %s\n",
                    g_logfile_path, strerror(errno));
        }
    }
    
    // Use mock dispatch handler for standalone mode
    jsonrpc_server_init(g_port, g_use_stdio, logfile, mock_dispatch_handler);
    
    jsonrpc_log("Starting standalone scenetree_passthrough");
    jsonrpc_log("===========================================");
    jsonrpc_log("Using MOCK handlers (no Godot API dependency)");
    
    // Print connection info for TCP mode
    if (!(g_use_stdio != false)) {
        printf("{\"event\":\"server_started\",\"host\":\"127.0.0.1\",\"port\":%d}\n", g_port);
        fflush(stdout);
    }
    
    // Run the server main loop
    jsonrpc_server_run();
    
    if ((logfile != NULL)) {
        fclose(logfile);
    }
    
    return 0;
}

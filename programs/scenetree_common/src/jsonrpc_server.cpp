// ==================== JSON-RPC 2.0 Server Implementation ====================
// Common JSON-RPC server implementation shared between modes.

#include "jsonrpc_server.hpp"
#include <stdarg.h>

// Global server state
JSONRPCServerState jsonrpc_server = {0};

// ==================== Server Initialization ====================

void jsonrpc_server_init(int port, bool use_stdio, FILE* logfile, JSONRPCHandlerFunc handler) {
    jsonrpc_server.port = port;
    jsonrpc_server.use_stdio = use_stdio;
    jsonrpc_server.sockfd = 0;
    jsonrpc_server.logfile = logfile;
    jsonrpc_server.handler = handler;

    if (!use_stdio && port > 0) {
        // TCP mode
        struct sockaddr_in address;
        int opt = 1;

        jsonrpc_server.sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (jsonrpc_server.sockfd < 0) {
            jsonrpc_log("Failed to create socket: %s", strerror(errno));
            return;
        }

        if (setsockopt(jsonrpc_server.sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
            jsonrpc_log("Warning: setsockopt failed: %s", strerror(errno));
        }

        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (bind(jsonrpc_server.sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            jsonrpc_log("Failed to bind to port %d: %s", port, strerror(errno));
            close(jsonrpc_server.sockfd);
            jsonrpc_server.sockfd = 0;
            return;
        }

        if (listen(jsonrpc_server.sockfd, 10) < 0) {
            jsonrpc_log("Failed to listen: %s", strerror(errno));
            close(jsonrpc_server.sockfd);
            jsonrpc_server.sockfd = 0;
            return;
        }

        jsonrpc_log("TCP server listening on port %d", port);
    } else {
        jsonrpc_log("Stdio mode enabled");
    }
}

// ==================== Request Parsing ====================

static const char* skip_whitespace(const char* p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static const char* parse_string(const char* p, char* out, size_t out_size) {
    if (*p != '"') return NULL;
    p++;

    char* dst = out;
    char* end = out + out_size - 1;

    while (*p && *p != '"') {
        if (*p == '\\') {
            p++;
            switch (*p) {
                case '"': *dst++ = '"'; break;
                case '\\': *dst++ = '\\'; break;
                case '/': *dst++ = '/'; break;
                case 'b': *dst++ = '\b'; break;
                case 'f': *dst++ = '\f'; break;
                case 'n': *dst++ = '\n'; break;
                case 'r': *dst++ = '\r'; break;
                case 't': *dst++ = '\t'; break;
                default:
                    if (*p == 'u') {
                        // Unicode escape - skip
                        p += 4;
                    }
                    break;
            }
            p++;
        } else {
            *dst++ = *p++;
        }

        if (dst >= end) {
            *dst = '\0';
            return NULL; // Buffer overflow
        }
    }

    if (*p == '"') {
        *dst = '\0';
        return p + 1;
    }

    return NULL; // Unterminated string
}

static const char* skip_value(const char* p) {
    p = skip_whitespace(p);

    switch (*p) {
        case '"':
            // String
            p = strchr(p + 1, '"');
            if (!p) return NULL;
            return p + 1;

        case '{':
            // Object
            {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    else if (*p == '"') {
                        p = strchr(p + 1, '"');
                        if (!p) return NULL;
                    }
                    p++;
                }
                if (depth != 0) return NULL;
                return p + 1;
            }

        case '[':
            // Array
            {
                int depth = 1;
                p++;
                while (*p && depth > 0) {
                    if (*p == '[') depth++;
                    else if (*p == ']') depth--;
                    else if (*p == '"') {
                        p = strchr(p + 1, '"');
                        if (!p) return NULL;
                    }
                    p++;
                }
                if (depth != 0) return NULL;
                return p + 1;
            }

        case 't':
            // true
            if (strncmp(p, "true", 4) == 0) return p + 4;
            return NULL;

        case 'f':
            // false
            if (strncmp(p, "false", 5) == 0) return p + 5;
            return NULL;

        case 'n':
            // null
            if (strncmp(p, "null", 4) == 0) return p + 4;
            return NULL;

        default:
            // Number
            if (*p == '-' || (*p >= '0' && *p <= '9')) {
                while (*p && (*p == '-' || *p == '+' || *p == 'e' || *p == 'E' ||
                             *p == '.' || (*p >= '0' && *p <= '9'))) p++;
                return p;
            }
            return NULL;
    }
}

int jsonrpc_parse_request(const char* request_text, JSONRPCRequest* out_request) {
    if (!request_text || !*request_text) return -1;

    // Skip whitespace
    const char* p = skip_whitespace(request_text);

    // Check for JSON-RPC 2.0 envelope
    if (strncmp(p, "{\"jsonrpc\"", 10) != 0) {
        return -1;
    }

    // Find method
    const char* method_start = strstr(p, "\"method\"");
    if (!method_start) {
        jsonrpc_log("No method found in request");
        return -1;
    }

    method_start += 8; // Skip "method"
    method_start = skip_whitespace(method_start);
    if (*method_start != ':') {
        jsonrpc_log("Invalid method format");
        return -1;
    }
    method_start = skip_whitespace(method_start + 1);

    if (*method_start != '"') {
        jsonrpc_log("Method must be a string");
        return -1;
    }

    char method[256];
    const char* str_end = parse_string(method_start, method, sizeof(method));
    if (!str_end) {
        jsonrpc_log("Invalid method string");
        return -1;
    }

    // Find id
    int id = 0;
    const char* id_start = strstr(p, "\"id\"");
    if (id_start) {
        id_start += 4; // Skip "id"
        id_start = skip_whitespace(id_start);
        if (*id_start == ':') {
            id_start = skip_whitespace(id_start + 1);
            if (*id_start == '"') {
                // String ID
                char id_str[64];
                const char* id_end = parse_string(id_start, id_str, sizeof(id_str));
                if (id_end) {
                    // Keep as string ID for response
                }
            } else if (*id_start == '-' || (*id_start >= '0' && *id_start <= '9')) {
                // Numeric ID
                id = atoi(id_start);
            }
        }
    }

    // Find params
    const char* params_start = strstr(p, "\"params\"");
    char params_json[JSONRPC_BUFFER_SIZE];
    
    if (params_start) {
        params_start += 8; // Skip "params"
        params_start = skip_whitespace(params_start);
        if (*params_start == ':') {
            params_start = skip_whitespace(params_start + 1);
            
            // Copy params JSON
            const char* params_end = skip_value(params_start);
            if (params_end) {
                size_t params_len = params_end - params_start;
                if (params_len >= sizeof(params_json)) {
                    jsonrpc_log("Params too large");
                    return -1;
                }
                strncpy(params_json, params_start, params_len);
                params_json[params_len] = '\0';
            } else {
                params_json[0] = '\0';
            }
        } else {
            params_json[0] = '\0';
        }
    } else {
        params_json[0] = '\0';
    }

    out_request->method = strdup(method);
    out_request->id = id;
    strncpy(out_request->params_json, params_json, sizeof(out_request->params_json) - 1);
    out_request->params_json[sizeof(out_request->params_json) - 1] = '\0';

    return 0;
}

// ==================== Response Helpers ====================

void jsonrpc_send_response(int id, const char* result_json) {
    char response[JSONRPC_BUFFER_SIZE];

    if (id == 0 && jsonrpc_server.use_stdio) {
        // Special case for notifications without ID
        snprintf(response, sizeof(response),
                 "{\"jsonrpc\":\"2.0\",\"result\":%s}\n",
                 result_json ? result_json : "null");
    } else {
        snprintf(response, sizeof(response),
                 "{\"jsonrpc\":\"2.0\",\"result\":%s,\"id\":%d}\n",
                 result_json ? result_json : "null",
                 id);
    }

    if (jsonrpc_server.sockfd > 0) {
        send(jsonrpc_server.sockfd, response, strlen(response), 0);
    } else if (jsonrpc_server.use_stdio) {
        printf("%s", response);
        fflush(stdout);
    }
}

void jsonrpc_send_error(int id, int code, const char* message, const char* data) {
    char response[JSONRPC_BUFFER_SIZE];
    char error_obj[512];

    if (data && *data) {
        snprintf(error_obj, sizeof(error_obj),
                 "{\"code\":%d,\"message\":\"%s\",\"data\":%s}",
                 code, message, data);
    } else {
        snprintf(error_obj, sizeof(error_obj),
                 "{\"code\":%d,\"message\":\"%s\"}",
                 code, message);
    }

    snprintf(response, sizeof(response),
             "{\"jsonrpc\":\"2.0\",\"error\":%s,\"id\":%d}\n",
             error_obj, id);

    if (jsonrpc_server.sockfd > 0) {
        send(jsonrpc_server.sockfd, response, strlen(response), 0);
    } else if (jsonrpc_server.use_stdio) {
        printf("%s", response);
        fflush(stdout);
    }
}

// ==================== Logging ====================

void jsonrpc_log(const char* format, ...) {
    if (!jsonrpc_server.logfile) return;

    va_list args;
    va_start(args, format);
    fprintf(jsonrpc_server.logfile, "[JSON-RPC] ");
    vfprintf(jsonrpc_server.logfile, format, args);
    fprintf(jsonrpc_server.logfile, "\n");
    fflush(jsonrpc_server.logfile);
    va_end(args);
}

// ==================== String Extraction Helpers ====================

const char* json_extract_string(const char* json, const char* key) {
    static char buffer[1024];
    const char* key_start = strstr(json, key);

    if (!key_start) return NULL;

    key_start = strchr(key_start + strlen(key), ':');
    if (!key_start) return NULL;
    key_start = skip_whitespace(key_start + 1);

    if (*key_start == '"') {
        const char* str_end = parse_string(key_start, buffer, sizeof(buffer));
        if (str_end) return buffer;
    }

    return NULL;
}

int json_extract_int(const char* json, const char* key, int default_value) {
    const char* val = json_extract_string(json, key);
    if (!val) return default_value;
    return atoi(val);
}

bool json_extract_bool(const char* json, const char* key, bool default_value) {
    const char* val = json_extract_string(json, key);
    if (!val) return default_value;
    if (strcmp(val, "true") == 0) return true;
    if (strcmp(val, "false") == 0) return false;
    return default_value;
}

// ==================== Stdio Mode Runner ====================

static void run_stdio_mode(void) {
    char buffer[JSONRPC_BUFFER_SIZE];

    while (fgets(buffer, sizeof(buffer), stdin)) {
        // Remove trailing newline
        char* newline = strchr(buffer, '\n');
        if (newline) *newline = '\0';

        JSONRPCRequest request;
        if (jsonrpc_parse_request(buffer, &request) == 0) {
            if (jsonrpc_server.handler) {
                const char* result = jsonrpc_server.handler(request.method, request.params_json, request.id);
                jsonrpc_send_response(request.id, result);
            }
            free((void*)request.method);
        } else {
            jsonrpc_send_error(0, JSONRPC_PARSE_ERROR, "Parse error", NULL);
        }
    }
}

// ==================== TCP Accept Loop ====================

static void* handle_client(void* arg) {
    int client_fd = *(int*)arg;
    free(arg);

    char buffer[JSONRPC_BUFFER_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = recv(client_fd, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_read] = '\0';
        jsonrpc_log("Received %zd bytes", bytes_read);

        JSONRPCRequest request;
        if (jsonrpc_parse_request(buffer, &request) == 0) {
            if (jsonrpc_server.handler) {
                const char* result = jsonrpc_server.handler(request.method, request.params_json, request.id);
                jsonrpc_send_response(request.id, result);
            }
            free((void*)request.method);
        } else {
            jsonrpc_send_error(0, JSONRPC_PARSE_ERROR, "Parse error", NULL);
        }

        memset(buffer, 0, sizeof(buffer));
    }

    close(client_fd);
    jsonrpc_log("Client disconnected");
    return NULL;
}

// ==================== Main Server Loop ====================

void jsonrpc_server_run(void) {
    if (jsonrpc_server.use_stdio) {
        run_stdio_mode();
        return;
    }

    if (jsonrpc_server.sockfd <= 0) {
        jsonrpc_log("No socket to run on");
        return;
    }

    jsonrpc_log("Waiting for connections...");

    while (1) {
        int client_fd;
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);

        client_fd = accept(jsonrpc_server.sockfd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            jsonrpc_log("Accept error: %s", strerror(errno));
            continue;
        }

        jsonrpc_log("Client connected from %s:%d",
                     inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));

        // Spawn handler thread
        pthread_t thread;
        int* client_ptr = (int*)malloc(sizeof(int));
        *client_ptr = client_fd;

        pthread_create(&thread, NULL, handle_client, client_ptr);
        pthread_detach(thread);
    }
}

#!/bin/bash
# ==================== Standalone Integration Tests ====================
#
# Usage: cd programs/scenetree_passthrough_standalone
#        ./test_standalone.sh

set -e

echo "==========================================="
echo "SceneTree Passthrough Standalone Tests"
echo "==========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

passed=0
failed=0

run_test() {
    local name="$1"
    local cmd="$2"
    local expected="$3"
    
    echo -n "Test: $name ... "
    
    local output
    if output=$(eval "$cmd" 2>&1); then
        if [[ "$output" == *"$expected"* ]]; then
            echo -e "${GREEN}PASS${NC}"
            ((passed++))
        else
            echo -e "${RED}FAIL${NC}"
            echo "  Expected: $expected"
            echo "  Got: $output"
            ((failed++))
        fi
    else
        echo -e "${RED}FAIL${NC}"
        ((failed++))
    fi
}

# Fix CMakeLists.txt to comment out GDNative target (needs Godot headers)
echo "Preparing build..."
pushd ../.. > /dev/null

# Create a minimal standalone build file that excludes GDNative
if [ ! -f .build_standalone_ready ]; then
    cmake -S programs/scenetree_passthrough_standalone -B .build_standalone 2>&1 | grep -v "^--" || true
    # Touch to mark ready
    touch .build_standalone_ready
fi

echo "Building..."
cd .build_standalone
# Comment out GDNative target lines
sed -i '/^add_library(scenetree_passthrough_gdnative/,/^)$/s/^/#/' CMakeLists.txt || true
sed -i '/install(TARGETS scenetree_passthrough_gdnative/,/^)$/s/^/#/' CMakeLists.txt || true
make 2>&1 | grep -v "^--" || true || true
echo "Done."
echo ""

# Run tests
echo "Running tests..."
echo ""

run_test "list_methods" \
    "./bin/scenetree_passthrough_standalone --stdio" \
    '"list_methods"'

run_test "get_tree" \
    "./bin/scenetree_passthrough_standalone --stdio" \
    '"active"'

OUTPUT=$(echo '{"jsonrpc":"2.0","method":"get_tree","id":42}' | ./bin/scenetree_passthrough_standalone --stdio)
if [[ "$OUTPUT" == *'"jsonrpc":"2.0"'* ]] && [[ "$OUTPUT" == *'"id":42'* ]]; then
    echo -e "${GREEN}PASS${NC}" "Response format"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}" "Response format"
    ((failed++))
fi

OUTPUT=$(echo '{"jsonrpc":"2.0","method":"nonexistent","id":1}' | ./bin/scenetree_passthrough_standalone --stdio 2>&1 || true)
if [[ "$OUTPUT" == *'"error"'* ]]; then
    echo -e "${GREEN}PASS${NC}" "Error handling"
    ((passed++))
else
    echo -e "${RED}FAIL${NC}" "Error handling"
    ((failed++))
fi

echo ""
echo "==========================================="
echo "Results: $passed passed, $failed failed"
echo "==========================================="

popd > /dev/null

[ $failed -eq 0 ]

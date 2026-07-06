#!/bin/bash
set -euo pipefail

PASS=0
FAIL=0
FCITX5_PID=""
FCITX5_LOG=""

pass() {
    echo "  PASS: $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

cleanup() {
    [ -n "$FCITX5_PID" ] && kill "$FCITX5_PID" 2>/dev/null || true
    [ -n "${DBUS_SESSION_BUS_PID:-}" ] && kill "$DBUS_SESSION_BUS_PID" 2>/dev/null || true
    [ -n "$FCITX5_LOG" ] && rm -f "$FCITX5_LOG"
}
trap cleanup EXIT

echo "=== Kashiko fcitx5 Integration Tests ==="
echo ""

# -------------------------------------------------------
# 1. Verify installed files exist
# -------------------------------------------------------
echo "[1/4] Checking installed files..."

ADDON_DIR=$(pkg-config --variable=libdir Fcitx5Core)/fcitx5
FCITX5_DATA="/usr/share/fcitx5"

if [ -f "$ADDON_DIR/kashiko.so" ]; then
    pass "kashiko.so installed at $ADDON_DIR/kashiko.so"
else
    fail "kashiko.so not found at $ADDON_DIR/kashiko.so"
fi

if [ -f "$ADDON_DIR/libkashiko_fcitx5.so" ]; then
    pass "libkashiko_fcitx5.so installed at $ADDON_DIR/libkashiko_fcitx5.so"
else
    fail "libkashiko_fcitx5.so not found at $ADDON_DIR/libkashiko_fcitx5.so"
fi

if [ -f "$FCITX5_DATA/addon/kashiko.conf" ]; then
    pass "addon config installed at $FCITX5_DATA/addon/kashiko.conf"
else
    fail "addon config not found at $FCITX5_DATA/addon/kashiko.conf"
fi

if [ -f "$FCITX5_DATA/inputmethod/kashiko.conf" ]; then
    pass "inputmethod config installed at $FCITX5_DATA/inputmethod/kashiko.conf"
else
    fail "inputmethod config not found at $FCITX5_DATA/inputmethod/kashiko.conf"
fi

# -------------------------------------------------------
# 2. Verify shared library linkage
# -------------------------------------------------------
echo ""
echo "[2/4] Checking shared library linkage..."

if ldd "$ADDON_DIR/kashiko.so" | grep -q "libkashiko_fcitx5.so"; then
    pass "kashiko.so links to libkashiko_fcitx5.so"
else
    fail "kashiko.so does not link to libkashiko_fcitx5.so"
fi

if ldd "$ADDON_DIR/kashiko.so" | grep -q "libFcitx5Core"; then
    pass "kashiko.so links to libFcitx5Core"
else
    fail "kashiko.so does not link to libFcitx5Core"
fi

# Verify RPATH is set to $ORIGIN so it can find libkashiko_fcitx5.so
RPATH_INFO=$(readelf -d "$ADDON_DIR/kashiko.so" 2>/dev/null | grep -E "RPATH|RUNPATH" || true)
if echo "$RPATH_INFO" | grep -q '\$ORIGIN'; then
    pass "kashiko.so has \$ORIGIN RPATH"
else
    fail "kashiko.so missing \$ORIGIN RPATH (got: ${RPATH_INFO:-none})"
fi

# -------------------------------------------------------
# 3. Verify addon config contents
# -------------------------------------------------------
echo ""
echo "[3/4] Checking addon configuration..."

ADDON_CONF="$FCITX5_DATA/addon/kashiko.conf"
if grep -q "Library=kashiko" "$ADDON_CONF"; then
    pass "addon config has Library=kashiko"
else
    fail "addon config missing Library=kashiko"
fi

if grep -q "Type=SharedLibrary" "$ADDON_CONF"; then
    pass "addon config has Type=SharedLibrary"
else
    fail "addon config missing Type=SharedLibrary"
fi

if grep -q "Category=InputMethod" "$ADDON_CONF"; then
    pass "addon config has Category=InputMethod"
else
    fail "addon config missing Category=InputMethod"
fi

IM_CONF="$FCITX5_DATA/inputmethod/kashiko.conf"
if grep -q "LangCode=ja" "$IM_CONF"; then
    pass "inputmethod config has LangCode=ja"
else
    fail "inputmethod config missing LangCode=ja"
fi

# -------------------------------------------------------
# 4. Start D-Bus and launch fcitx5, verify addon loading
# -------------------------------------------------------
echo ""
echo "[4/4] Starting fcitx5 and checking addon loading..."

# Start D-Bus session bus
eval "$(dbus-launch --sh-syntax)"
export DBUS_SESSION_BUS_ADDRESS

# Launch fcitx5 in the background, capture log
# Note: do NOT use -d (daemon) flag as it forks and loses stderr redirection
FCITX5_LOG=$(mktemp)
fcitx5 --verbose '*=5' >"$FCITX5_LOG" 2>&1 &
FCITX5_PID=$!

# Wait for fcitx5 to start (up to 15 seconds)
STARTED=false
for i in $(seq 1 30); do
    sleep 0.5
    if grep -q "Loaded addon kashiko" "$FCITX5_LOG" 2>/dev/null; then
        STARTED=true
        break
    fi
    # Also check if fcitx5 has exited unexpectedly
    if ! kill -0 "$FCITX5_PID" 2>/dev/null; then
        break
    fi
done

if [ "$STARTED" = true ]; then
    pass "fcitx5 loaded addon kashiko successfully"
else
    fail "fcitx5 did not load addon kashiko within timeout"
    echo "  --- fcitx5 log (last 50 lines) ---"
    tail -50 "$FCITX5_LOG" | sed 's/^/  | /'
    echo "  --- end log ---"
fi

# Check that no load error occurred for kashiko
if grep -q "Could not load addon kashiko" "$FCITX5_LOG" 2>/dev/null; then
    fail "fcitx5 reported error loading kashiko addon"
    grep "kashiko" "$FCITX5_LOG" | sed 's/^/  | /'
else
    pass "no load error for kashiko addon"
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

echo ""
echo "All integration tests passed!"
exit 0

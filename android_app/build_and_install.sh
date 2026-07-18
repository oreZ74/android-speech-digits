#!/bin/bash

# Android Build Script
# Builds debug APK and installs on connected device

set -e

cd "$(dirname "$0")"

echo "======================================"
echo "  Android Build"
echo "======================================"
echo ""

# 1. Clean Build
echo "[1/4] Cleaning previous builds..."
./gradlew clean

# 2. Assemble Debug APK
echo "[2/4] Building Debug APK..."
./gradlew assembleDebug

# 3. Check if APK exists
APK_PATH="app/build/outputs/apk/debug/app-debug.apk"
if [ ! -f "$APK_PATH" ]; then
    echo "[ERROR] APK not found at $APK_PATH"
    exit 1
fi

echo "[OK] APK built successfully:"
ls -lh "$APK_PATH"

# 4. Install on device (if connected)
echo "[3/4] Checking for connected devices..."
if adb devices | grep -q "device$"; then
    echo "[4/4] Installing APK on device..."
    adb install -r "$APK_PATH"
    echo ""
    echo "[OK] Installation complete!"
    echo ""
    echo "To view logs:"
    echo "  adb logcat -s RawDigitClassifier RawDigitDebug MainActivity"
else
    echo "[WARNING] No device connected. Skipping installation."
    echo ""
    echo "To install manually:"
    echo "  adb install -r $APK_PATH"
fi

echo ""
echo "======================================"
echo "  Build Complete!"
echo "======================================"

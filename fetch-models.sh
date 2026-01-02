#!/bin/bash
set -e

# crimson-mute model fetcher
# Downloads AMD ANR ONNX models from AMD's Adrenalin driver package

AMD_DRIVER_URL="https://drivers.amd.com/drivers/whql-amd-software-adrenalin-edition-25.12.1-win11-b.exe"
AMD_EULA_URL="https://www.amd.com/en/legal/eula/amd-software-eula.html"
AMD_REFERER="https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-rx/radeon-rx-9000-series/amd-radeon-rx-9070-xt.html"
USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
DRIVER_FILENAME="whql-amd-software-adrenalin-edition-25.12.1-win11-b.exe"
ONNX_PATH_IN_ARCHIVE="Packages/Drivers/Display/WT6A_INF/amdfdans"

# Well-known location for Linux
get_well_known_dir() {
    if [ -n "$XDG_DATA_HOME" ]; then
        echo "$XDG_DATA_HOME/anr-plugin"
    else
        echo "$HOME/.local/share/anr-plugin"
    fi
}

WELL_KNOWN_DIR=$(get_well_known_dir)

# Check for required utilities upfront
if command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
elif command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
else
    echo "Error: Neither curl nor wget found. Please install one of them."
    exit 1
fi

if command -v 7z &> /dev/null; then
    EXTRACT_CMD="7z"
elif command -v 7zr &> /dev/null; then
    EXTRACT_CMD="7zr"
elif command -v 7za &> /dev/null; then
    EXTRACT_CMD="7za"
else
    echo "Error: 7-zip not found. Please install p7zip (p7zip-full on Debian/Ubuntu)."
    echo "  Ubuntu/Debian: sudo apt install p7zip-full"
    echo "  Fedora: sudo dnf install p7zip p7zip-plugins"
    echo "  Arch: sudo pacman -S p7zip"
    exit 1
fi

echo "-------------------------------------------------------------------------"
echo "                          DISCLAIMER & AGREEMENT"
echo "-------------------------------------------------------------------------"
echo "This tool downloads and extracts components from the official AMD Drivers."
echo "By proceeding, you acknowledge that:"
echo ""
echo "1. You own AMD hardware and are entitled to use this software."
echo "2. You accept the terms of the AMD EULA:"
echo "   $AMD_EULA_URL"
echo "3. You take full responsibility for using these extracted components."
echo ""
echo "This software is provided \"as is\" without warranty. The author is not"
echo "liable for any license violations or your computer exploding."
echo "-------------------------------------------------------------------------"
echo ""

read -p "Do you accept the EULA and wish to proceed? [y/N]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Where would you like to install the models?"
echo ""
echo "  1) Well-known location: $WELL_KNOWN_DIR (recommended)"
echo "  2) Current directory: $(pwd)"
echo ""
read -p "Choice [1]: " -n 1 -r INSTALL_CHOICE
echo ""

if [ "$INSTALL_CHOICE" = "2" ]; then
    INSTALL_DIR="$(pwd)"
else
    INSTALL_DIR="$WELL_KNOWN_DIR"
fi

echo ""
echo "Models will be installed to: $INSTALL_DIR"
echo ""

# Create temp directory
TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT

echo "Downloading AMD driver package..."
echo "(This is a large file, ~1GB, please be patient)"
echo ""

if [ "$DOWNLOAD_CMD" = "curl" ]; then
    curl --http1.1 -L -o "$TMPDIR/$DRIVER_FILENAME" "$AMD_DRIVER_URL" \
        -A "$USER_AGENT" -e "$AMD_REFERER"
else
    wget -O "$TMPDIR/$DRIVER_FILENAME" "$AMD_DRIVER_URL" \
        -U "$USER_AGENT" --referer="$AMD_REFERER"
fi

echo ""
echo "Extracting ONNX models..."

# Extract just the ONNX files from the archive
cd "$TMPDIR"
$EXTRACT_CMD x -y "$DRIVER_FILENAME" "$ONNX_PATH_IN_ARCHIVE/*.onnx" > /dev/null

# Check if extraction succeeded
if [ ! -d "$TMPDIR/$ONNX_PATH_IN_ARCHIVE" ]; then
    echo "Error: Could not find ONNX models in archive."
    echo "The archive structure may have changed. Please report this issue."
    exit 1
fi

# Count models found
MODEL_COUNT=$(find "$TMPDIR/$ONNX_PATH_IN_ARCHIVE" -name "*.onnx" | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    echo "Error: No ONNX models found in the expected location."
    exit 1
fi

echo "Found $MODEL_COUNT ONNX model(s)"

# Create install directory if needed
mkdir -p "$INSTALL_DIR"

# Copy models
cp "$TMPDIR/$ONNX_PATH_IN_ARCHIVE"/*.onnx "$INSTALL_DIR/"

echo ""
echo "=============================================="
echo "  Installation complete!"
echo "=============================================="
echo ""
echo "Models installed to: $INSTALL_DIR"
echo ""
ls -lh "$INSTALL_DIR"/*.onnx

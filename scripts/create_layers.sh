#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Define variables
LAYER_BASE_DIR="layers"
LAYERS=(
    "milvus"
    "clusters"
)

# Function to clean up previous builds
cleanup_layer() {
    local layer_name=$1
    local layer_dir="${LAYER_BASE_DIR}/python-requirements-${layer_name}"
    local output_zip="${LAYER_BASE_DIR}/python-requirements-${layer_name}.zip"

    echo "Cleaning up previous builds for ${layer_name}..."
    rm -rf "$layer_dir"
    rm -f "$output_zip"
}

# Function to export Poetry requirements
export_requirements() {
    local layer_name=$1
    local requirements_file="${LAYER_BASE_DIR}/requirements-${layer_name}.txt"

    echo "Exporting requirements for ${layer_name}..."
    poetry export --without-hashes --with "$layer_name" -o "$requirements_file"
}

# Function to install dependencies into the layer directory
install_dependencies() {
    local layer_name=$1
    local requirements_file="${LAYER_BASE_DIR}/requirements-${layer_name}.txt"
    local layer_dir="${LAYER_BASE_DIR}/python-requirements-${layer_name}/python"

    echo "Installing dependencies for ${layer_name}..."
    mkdir -p "$layer_dir"
    pip install -r "$requirements_file" -t "$layer_dir"
}

# Function to create the ZIP file for the layer
zip_layer() {
    local layer_name=$1
    local layer_dir="${LAYER_BASE_DIR}/python-requirements-${layer_name}"
    local output_zip="${LAYER_BASE_DIR}/python-requirements-${layer_name}.zip"

    echo "Zipping dependencies for ${layer_name}..."
    cd "$layer_dir" && zip -r "../python-requirements-${layer_name}.zip" python/ && cd -
    echo "Layer ZIP created: $output_zip"
}

# Main script execution
echo "Starting layer creation process..."

# Ensure Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed. Please install Poetry and try again."
    exit 1
fi

# Ensure pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install pip and try again."
    exit 1
fi

# Loop through each layer and perform all required steps
for layer_name in "${LAYERS[@]}"; do
    echo "Processing layer: ${layer_name}"
    cleanup_layer "$layer_name"
    export_requirements "$layer_name"
    install_dependencies "$layer_name"
    zip_layer "$layer_name"
    echo "Layer ${layer_name} completed successfully."
done

echo "All layers have been created successfully!"

#!/bin/sh

# Export environment variables for Metal/macOS development
export MACOSX_DEPLOYMENT_TARGET=10.15

# Set Metal-specific environment variables if on macOS
if [ "$(uname)" = "Darwin" ]; then
    # Ensure we can find Metal frameworks
    export FRAMEWORK_PATH="/System/Library/Frameworks"
    
    # Set Metal compiler path (usually handled automatically)
    if [ -z "$METAL_COMPILER_PATH" ]; then
        export METAL_COMPILER_PATH="$(xcrun --find metal)"
    fi
    
    # Ensure proper SDK path for Metal development
    if [ -z "$SDKROOT" ]; then
        export SDKROOT="$(xcrun --show-sdk-path)"
    fi
    
    echo "macOS Metal development environment:"
    echo "  MACOSX_DEPLOYMENT_TARGET: $MACOSX_DEPLOYMENT_TARGET"
    echo "  SDKROOT: $SDKROOT"
    echo "  Metal compiler: $METAL_COMPILER_PATH"
fi

echo "Building btrack..."

# make compilation directories
mkdir -p ./btrack/libs
mkdir -p ./btrack/obj

# clone Eigen if not present
if [ ! -e ./btrack/include/eigen/signature_of_eigen3_matrix_library ]
then
    echo "Cloning Eigen library..."
    git clone --depth 1 --branch 3.3.9 https://gitlab.com/libeigen/eigen.git ./btrack/include/eigen
fi

# Show build configuration
echo "Checking build configuration..."
make info

# build the tracker
echo "Compiling btrack from source..."
make clean
make

# Verify build results
if [ -f "./btrack/libs/libtracker.dylib" ] || [ -f "./btrack/libs/libtracker.so" ]; then
    echo "✅ Build successful!"
    
    # Show what was built
    echo "Built files:"
    ls -la ./btrack/libs/
    
    # Check for Metal shaders on macOS
    if [ "$(uname)" = "Darwin" ] && [ -f "./btrack/libs/belief.metal" ]; then
        echo "✅ Metal shaders copied successfully"
    fi
    
    # Check for object files
    echo "Object files:"
    ls -la ./btrack/obj/
    
else
    echo "❌ Build failed!"
    exit 1
fi

echo "Build complete!"

#!/bin/sh
echo "Building btrack..."
# make compilation directories
mkdir ./btrack/libs
mkdir ./btrack/obj

# build the tracker
echo "Compiling btrack from source..."
make clean
make

# run the installation
echo "Installing btrack python package..."
pip install -e .

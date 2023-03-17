#!/bin/sh
echo "Building btrack..."
# make compilation directories
mkdir ./btrack/libs
mkdir ./btrack/obj

# clone Eigen
if [ ! -e ./btrack/include/eigen/signature_of_eigen3_matrix_library ]
then
  git clone https://github.com/eigen-mirror/eigen ./btrack/include/eigen
fi

# build the tracker
echo "Compiling btrack from source..."
make clean
make

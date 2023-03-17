#!/bin/sh
echo "Building btrack..."
# make compilation directories
mkdir ./btrack/libs
mkdir ./btrack/obj

# clone Eigen
if [ ! -e ./btrack/include/eigen/signature_of_eigen3_matrix_library ]
then
  git clone https://gitlab.com/libeigen/eigen.git ./btrack/include/eigen
fi

UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	# do something Linux #-fopenmp -static
	CXX = g++
	EXT = so
	XLDFLAGS = -Wl,--no-undefined -Wl,--no-allow-shlib-undefined
	#-L/usr/local/cuda/lib64 -lcuda -lcudart
endif
ifeq ($(UNAME), Darwin)
	# do something OSX
	CXX = clang++
	EXT = dylib
endif

NVCC = nvcc


# If your compiler is a bit older you may need to change -std=c++11 to -std=c++0x
#-I/usr/include/python2.7 -L/usr/lib/python2.7 # -O3
GDBFLAGS = -g3 -O0 -ggdb
CXXFLAGS = -Wall -c -std=c++11 -m64 -O3 -fPIC -DDEBUG=false -DFAST_COST_UPDATE=false -Iinclude
LDFLAGS = -shared $(XLDFLAGS)

EXE = tracker
SRC_DIR = ./src
OBJ_DIR = ./obj
SRC = $(wildcard $(SRC_DIR)/*.cc)
OBJ = $(SRC:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)



all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(LDFLAGS) -o ./libs/libtracker.$(EXT) $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJ)

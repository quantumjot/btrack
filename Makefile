UNAME := $(shell uname)

CXX = x86_64-w64-mingw32-g++
EXT = DLL
XLDFLAGS = -static-libgcc -static-libstdc++

NVCC = nvcc


# automatically get the version numbers from VERSION.txt
VERSION_FILE := ./btrack/VERSION.txt
VERSION_MAJOR = $(shell cat $(VERSION_FILE) | cut -f1 -d.)
VERSION_MINOR = $(shell cat $(VERSION_FILE) | cut -f2 -d.)
VERSION_BUILD = $(shell cat $(VERSION_FILE) | cut -f3 -d.)

# If your compiler is a bit older you may need to change -std=c++17 to -std=c++0x
#-I/usr/include/python2.7 -L/usr/lib/python2.7 # -O3
LLDBFLAGS =
CXXFLAGS = -c -std=c++17 -m64 -fPIC -I"./btrack/include" \
					 -DDEBUG=false -DVERSION_MAJOR=$(VERSION_MAJOR) \
					 -DVERSION_MINOR=$(VERSION_MINOR) -DVERSION_BUILD=$(VERSION_BUILD) \
					 -DBUILD_SHARED_LIB
OPTFLAGS = -O3
LDFLAGS = -shared $(XLDFLAGS)


EXE = tracker
SRC_DIR = ./btrack/src
OBJ_DIR = ./btrack/obj
SRC = $(wildcard $(SRC_DIR)/*.cc)
OBJ = $(SRC:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

# make it
all: $(EXE)

$(EXE): $(OBJ)
	$(CXX) $(LDFLAGS) -o ./btrack/libs/libtracker.$(EXT) $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(LLDBFLAGS) -c $< -o $@

# make with debug symbols
debug: LLDBFLAGS = -glldb
debug: OPTFLAGS =
debug: all

clean:
	$(RM) $(OBJ)

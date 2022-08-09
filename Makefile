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
	CXX = clang++ -arch x86_64 -arch arm64
	EXT = dylib
	XLD_FLAGS = -arch x86_64 -arch arm64
endif
ifeq ($(UNAME), Windows)
	# do something Windowsy
	CXX = x86_64-w64-mingw32-g++
	EXT = DLL
	XLDFLAGS = -static-libgcc -static-libstdc++
endif

NVCC = nvcc


# automatically get the version numbers from VERSION.txt
VERSION_FILE := ./btrack/VERSION.txt
VERSION_MAJOR = $(shell cat $(VERSION_FILE) | cut -f1 -d.)
VERSION_MINOR = $(shell cat $(VERSION_FILE) | cut -f2 -d.)
VERSION_BUILD = $(shell cat $(VERSION_FILE) | cut -f3 -d.)

# If your compiler is a bit older you may need to change -std=c++11 to -std=c++0x
#-I/usr/include/python2.7 -L/usr/lib/python2.7 # -O3
LLDBFLAGS =
CXXFLAGS = -c -std=c++11 -m64 -fPIC -I"./btrack/include" \
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

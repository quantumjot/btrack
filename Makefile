UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
	ifeq ($(WINDOWS_CROSS_COMPILE),true)
		# do something Windowsy
		CXX = x86_64-w64-mingw32-g++
		EXT	= DLL
		XLDFLAGS = -static-libgcc -static-libstdc++
	else
		# do something Linux #-fopenmp -static
		CXX = g++ -lstdc++fs
		EXT = so
		XLDFLAGS = -Wl,--no-undefined -Wl,--no-allow-shlib-undefined
		#-L/usr/local/cuda/lib64 -lcuda -lcudart
	endif
endif
ifeq ($(UNAME), Darwin)
	# do something OSX
	CXX = clang++ -arch x86_64 -arch arm64
	EXT = dylib
	XLDFLAGS =
endif

NVCC = nvcc

# If your compiler is a bit older you may need to change -std=c++17 to -std=c++0x
#-I/usr/include/python2.7 -L/usr/lib/python2.7 # -O3
LLDBFLAGS =
CXXFLAGS = -c -std=gnu++17 -m64 -fPIC -I"./btrack/include" \
	       -DDEBUG=false -DBUILD_SHARED_LIB
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

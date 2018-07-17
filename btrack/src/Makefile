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
CXXFLAGS = -Wall -c -std=c++11 -m64 -O3 -fPIC -DDEBUG=false -I"../include/"
LDFLAGS = -shared $(XLDFLAGS)

EXE = tracker
OBJ = motion.o inference.o tracklet.o hyperbin.o hypothesis.o manager.o tracker.o wrapper.o interface.o
DEPS = types.h motion.h inference.h tracklet.h hyperbin.h tracker.h hypothesis.h manager.h wrapper.h interface.h

all: $(EXE)

# belief.o: belief.cu
# 	$(NVCC) -std=c++11 --compiler-options '-fPIC' -m64 -O3 $(INCLUDEFLAGS) -dc belief.cu -o belief.o

$(EXE): $(OBJ)
	$(CXX) $(LDFLAGS) -o ../libs/libtracker.$(EXT) $^

%.o: %.c $(DEPS)
	$(CXX) $(INCLUDEFLAGS) $(CXXFLAGS) $< -o $@



clean:
	rm *.o #&& rm $(EXE)

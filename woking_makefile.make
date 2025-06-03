UNAME := $(shell uname)

# Set RM explicitly
RM = rm -f

# CUDA environment for conda installation
ifeq ($(CONDA_PREFIX),)
    CUDA_HOME = /usr/local/cuda
else
    CUDA_HOME = $(CONDA_PREFIX)
endif

ifeq ($(UNAME), Linux)
    ifeq ($(WINDOWS_CROSS_COMPILE),true)
        CXX = x86_64-w64-mingw32-g++
        EXT	= DLL
        XLDFLAGS = -static-libgcc -static-libstdc++
    else
        CXX = g++
        EXT = so
        XLDFLAGS = -Wl,--no-undefined -Wl,--no-allow-shlib-undefined
    endif
endif
ifeq ($(UNAME), Darwin)
    CXX = clang++ -arch x86_64 -arch arm64
    EXT = dylib
    XLDFLAGS =
    # Objective-C++ compiler for Metal files
    OBJCXX = clang++ -arch x86_64 -arch arm64
endif

NVCC = $(CUDA_HOME)/bin/nvcc

# Compiler flags
CXXFLAGS = -c -std=c++11 -m64 -fPIC -I"./btrack/include" \
           -DDEBUG=false -DBUILD_SHARED_LIB

# Add availability defines based on detection
ifeq ($(CUDA_AVAILABLE),yes)
    CXXFLAGS += -DHAVE_CUDA
endif

ifeq ($(METAL_AVAILABLE),yes)
    CXXFLAGS += -DHAVE_METAL
    OBJCXXFLAGS += -DHAVE_METAL
endif

OPTFLAGS = -O3
LDFLAGS = -shared $(XLDFLAGS)

# Objective-C++ flags for Metal compilation
OBJCXXFLAGS = -c -std=c++11 -m64 -fPIC -I"./btrack/include" \
              -DDEBUG=false -DBUILD_SHARED_LIB -fobjc-arc

EXE = tracker
SRC_DIR = ./btrack/src
OBJ_DIR = ./btrack/obj
LIBS_DIR = ./btrack/libs

# Source files
SRC = $(wildcard $(SRC_DIR)/*.cc)
CUDA_SRC = $(wildcard $(SRC_DIR)/*.cu)
METAL_SRC = $(wildcard $(SRC_DIR)/*.mm)
METAL_SHADERS = $(wildcard $(SRC_DIR)/*.metal)

# Object files
OBJ = $(SRC:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CUDA_OBJ = $(CUDA_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
METAL_OBJ = $(METAL_SRC:$(SRC_DIR)/%.mm=$(OBJ_DIR)/%.o)

# Check if CUDA is available
CUDA_AVAILABLE := $(shell test -f $(CUDA_HOME)/bin/nvcc && echo yes || echo no)

# Check if Metal is available (macOS only)
ifeq ($(UNAME), Darwin)
    METAL_AVAILABLE := $(shell test -d /System/Library/Frameworks/Metal.framework && echo yes || echo no)
else
    METAL_AVAILABLE := no
endif

# Determine which objects to build
ALL_OBJ = $(OBJ)

ifeq ($(CUDA_AVAILABLE),yes)
    ALL_OBJ += $(CUDA_OBJ)
endif

ifeq ($(METAL_AVAILABLE),yes)
    ALL_OBJ += $(METAL_OBJ)
endif

all: $(EXE)

$(EXE): $(ALL_OBJ)
	@mkdir -p $(LIBS_DIR)
ifeq ($(UNAME), Darwin)
    ifeq ($(METAL_AVAILABLE),yes)
        ifeq ($(CUDA_AVAILABLE),yes)
		$(CXX) $(LDFLAGS) -framework Metal -framework Foundation -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
        else
		$(CXX) $(LDFLAGS) -framework Metal -framework Foundation -o $(LIBS_DIR)/libtracker.$(EXT) $^
        endif
		@echo "Copying Metal shaders..."
		@for shader in $(METAL_SHADERS); do \
			cp $$shader $(LIBS_DIR)/; \
		done
    else
        ifeq ($(CUDA_AVAILABLE),yes)
		$(CXX) $(LDFLAGS) -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
        else
		$(CXX) $(LDFLAGS) -o $(LIBS_DIR)/libtracker.$(EXT) $^
        endif
    endif
else
    ifeq ($(CUDA_AVAILABLE),yes)
	$(CXX) $(LDFLAGS) -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
    else
	$(CXX) $(LDFLAGS) -o $(LIBS_DIR)/libtracker.$(EXT) $^
    endif
endif

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
ifeq ($(CUDA_AVAILABLE),yes)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c -std=c++11 -m64 -Xcompiler -fPIC -I"./btrack/include" -o $@ $<
else
	@echo "CUDA not available, skipping $<"
endif

# Compile Objective-C++ source files (Metal)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm
ifeq ($(METAL_AVAILABLE),yes)
	@mkdir -p $(OBJ_DIR)
	$(OBJCXX) $(OBJCXXFLAGS) $(OPTFLAGS) -c $< -o $@
else
	@echo "Metal not available, skipping $<"
endif

debug: OPTFLAGS =
debug: allUNAME := $(shell uname)

# Set RM explicitly
RM = rm -f

# CUDA environment for conda installation
ifeq ($(CONDA_PREFIX),)
    CUDA_HOME = /usr/local/cuda
else
    CUDA_HOME = $(CONDA_PREFIX)
endif

ifeq ($(UNAME), Linux)
    ifeq ($(WINDOWS_CROSS_COMPILE),true)
        CXX = x86_64-w64-mingw32-g++
        EXT	= DLL
        XLDFLAGS = -static-libgcc -static-libstdc++
    else
        CXX = g++
        EXT = so
        XLDFLAGS = -Wl,--no-undefined -Wl,--no-allow-shlib-undefined
    endif
endif
ifeq ($(UNAME), Darwin)
    CXX = clang++ -arch x86_64 -arch arm64
    EXT = dylib
    XLDFLAGS =
    # Objective-C++ compiler for Metal files
    OBJCXX = clang++ -arch x86_64 -arch arm64
endif

NVCC = $(CUDA_HOME)/bin/nvcc

# Compiler flags
CXXFLAGS = -c -std=c++11 -m64 -fPIC -I"./btrack/include" \
           -DDEBUG=false -DBUILD_SHARED_LIB

# Add availability defines based on detection
CUDA_AVAILABLE := $(shell test -f $(CUDA_HOME)/bin/nvcc && echo yes || echo no)

# Check if Metal is available (macOS only)
ifeq ($(UNAME), Darwin)
    METAL_AVAILABLE := $(shell test -d /System/Library/Frameworks/Metal.framework && echo yes || echo no)
else
    METAL_AVAILABLE := no
endif

ifeq ($(CUDA_AVAILABLE),yes)
    CXXFLAGS += -DHAVE_CUDA
endif

ifeq ($(METAL_AVAILABLE),yes)
    CXXFLAGS += -DHAVE_METAL
endif

OPTFLAGS = -O3
LDFLAGS = -shared $(XLDFLAGS)

# Objective-C++ flags for Metal compilation  
OBJCXXFLAGS = -c -std=c++11 -m64 -fPIC -I"./btrack/include" \
              -DDEBUG=false -DBUILD_SHARED_LIB -fobjc-arc

ifeq ($(METAL_AVAILABLE),yes)
    OBJCXXFLAGS += -DHAVE_METAL
endif

EXE = tracker
SRC_DIR = ./btrack/src
OBJ_DIR = ./btrack/obj
LIBS_DIR = ./btrack/libs

# Source files
SRC = $(wildcard $(SRC_DIR)/*.cc)
CUDA_SRC = $(wildcard $(SRC_DIR)/*.cu)
METAL_SRC = $(wildcard $(SRC_DIR)/*.mm)
METAL_SHADERS = $(wildcard $(SRC_DIR)/*.metal)

# Object files
OBJ = $(SRC:$(SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)
CUDA_OBJ = $(CUDA_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
METAL_OBJ = $(METAL_SRC:$(SRC_DIR)/%.mm=$(OBJ_DIR)/%.o)

# Determine which objects to build
ALL_OBJ = $(OBJ)

ifeq ($(CUDA_AVAILABLE),yes)
    ALL_OBJ += $(CUDA_OBJ)
endif

ifeq ($(METAL_AVAILABLE),yes)
    ALL_OBJ += $(METAL_OBJ)
endif

all: $(EXE)

$(EXE): $(ALL_OBJ)
	@mkdir -p $(LIBS_DIR)
ifeq ($(UNAME), Darwin)
    ifeq ($(METAL_AVAILABLE),yes)
        ifeq ($(CUDA_AVAILABLE),yes)
		$(CXX) $(LDFLAGS) -framework Metal -framework Foundation -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
        else
		$(CXX) $(LDFLAGS) -framework Metal -framework Foundation -o $(LIBS_DIR)/libtracker.$(EXT) $^
        endif
		@echo "Copying Metal shaders..."
		@for shader in $(METAL_SHADERS); do \
			cp $$shader $(LIBS_DIR)/; \
		done
    else
        ifeq ($(CUDA_AVAILABLE),yes)
		$(CXX) $(LDFLAGS) -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
        else
		$(CXX) $(LDFLAGS) -o $(LIBS_DIR)/libtracker.$(EXT) $^
        endif
    endif
else
    ifeq ($(CUDA_AVAILABLE),yes)
	$(CXX) $(LDFLAGS) -lcudart -L$(CUDA_HOME)/lib64 -o $(LIBS_DIR)/libtracker.$(EXT) $^
    else
	$(CXX) $(LDFLAGS) -o $(LIBS_DIR)/libtracker.$(EXT) $^
    endif
endif

# Compile C++ source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

# FIXED: Conditional Metal compilation rules (separate rules, not conditional inside rule)
ifeq ($(METAL_AVAILABLE),yes)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm
	@mkdir -p $(OBJ_DIR)
	@echo "Compiling Metal source: $< → $@"
	$(OBJCXX) $(OBJCXXFLAGS) $(OPTFLAGS) -c $< -o $@
	@echo "✅ Metal object created: $@"
else
# If Metal not available, create dummy rule that shows skip message
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.mm
	@echo "Metal not available, skipping $<"
endif

# Compile CUDA source files
ifeq ($(CUDA_AVAILABLE),yes)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) -c -std=c++11 -m64 -Xcompiler -fPIC -I"./btrack/include" -o $@ $<
else
# If CUDA not available, create dummy rule that shows skip message  
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "CUDA not available, skipping $<"
endif

debug: OPTFLAGS =
debug: all

clean:
	$(RM) $(OBJ) $(CUDA_OBJ) $(METAL_OBJ)
	$(RM) $(LIBS_DIR)/*.metal

info:
	@echo "Build configuration:"
	@echo "  Platform: $(UNAME)"
	@echo "  CUDA available: $(CUDA_AVAILABLE)"
	@echo "  Metal available: $(METAL_AVAILABLE)"
	@echo "  C++ compiler: $(CXX)"
ifeq ($(UNAME), Darwin)
	@echo "  Objective-C++ compiler: $(OBJCXX)"
endif
	@echo "  Sources: $(words $(SRC)) C++, $(words $(CUDA_SRC)) CUDA, $(words $(METAL_SRC)) Metal"
	@echo "  Objects to build: $(words $(ALL_OBJ))"

.PHONY: all debug clean info


clean:
	$(RM) $(ALL_OBJ)
	$(RM) $(LIBS_DIR)/*.metal

info:
	@echo "Build configuration:"
	@echo "  Platform: $(UNAME)"
	@echo "  CUDA available: $(CUDA_AVAILABLE)"
	@echo "  Metal available: $(METAL_AVAILABLE)"
	@echo "  C++ compiler: $(CXX)"
ifeq ($(UNAME), Darwin)
	@echo "  Objective-C++ compiler: $(OBJCXX)"
endif
	@echo "  Sources: $(words $(SRC)) C++, $(words $(CUDA_SRC)) CUDA, $(words $(METAL_SRC)) Metal"
	@echo "  Objects to build: $(words $(ALL_OBJ))"

.PHONY: all debug clean info

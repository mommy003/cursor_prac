# Same layout as /Users/uqjzeng1/XCode/GCTB/scr/Makefile (Eigen/Boost paths, LLVM clang++ on Darwin, libomp).
# Plus: eigen.cpp, quantizer.cpp, stratifyMixture.cpp omitted (not in upstream scr/Makefile).
#
# Override defaults:
#   make EIGEN3_INCLUDE_DIR=/path/to/eigen BOOST_LIB=/path/to/boost

OUTPUT = gctb

EIGEN3_INCLUDE_DIR ?= /Users/uqjzeng1/XCode/eigen-3.4.0
BOOST_LIB ?= /Users/uqjzeng1/XCode/boost_1_81_0

EIGEN_PATH = $(EIGEN3_INCLUDE_DIR)
BOOST_PATH = $(BOOST_LIB)

# quantizer.hpp uses std::filesystem — needs C++17 (upstream scr uses c++0x; we bump only here).
ifeq ($(shell uname -s),Linux)
    CXX = g++
    CXXFLAGS = -I $(EIGEN_PATH) -I $(BOOST_PATH) -DUNIX -DNDEBUG -msse2 -m64 -fopenmp -O3 -std=c++17 -Wall -w
    LIB = -ldl -lz -fopenmp
else ifeq ($(shell uname -s),Darwin)
    ifneq ("$(wildcard /opt/homebrew/opt/llvm/bin/clang++)","")
        CXX = /opt/homebrew/opt/llvm/bin/clang++
    else
        CXX = clang++
    endif
    CXXFLAGS = -I /usr/local/include -I $(EIGEN_PATH) -I $(BOOST_PATH) -DUNIX -Dfopen64=fopen -stdlib=libc++ -m64 -fopenmp -O3 -std=c++17 -Wall -w
    LIB = -lz -lm -L/usr/local/opt/libomp/lib -L/opt/homebrew/opt/libomp/lib
endif

HDR = gctb.hpp data.hpp gadgets.hpp hsq.hpp mcmc.hpp model.hpp options.hpp stat.hpp vgmaf.hpp predict.hpp xci.hpp stratify.hpp multichain.hpp quantizer.hpp
SRC = gctb.cpp data.cpp gadgets.cpp hsq.cpp main.cpp mcmc.cpp model.cpp options.cpp stat.cpp vgmaf.cpp predict.cpp xci.cpp stratify.cpp eigen.cpp multichain.cpp quantizer.cpp
OBJ = $(SRC:.cpp=.o)

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(OUTPUT) $(OBJ) $(LIB)

%.o: %.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(OUTPUT) *~

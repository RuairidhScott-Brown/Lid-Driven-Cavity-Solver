# Compiler
CXX := mpicxx
CC := mpicxx

# Compiler flags
CXXFLAGS := -std=c++11 -Wall -pedantic -fopenmp -O3
TEST_CXXFLAGS := -std=c++11 -Wall -fopenmp -O3

# Directories
SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build
DOC_DIR := doc
TESTS_DIR := tests
BIN_DIR := bin

# Libraries
LIBS := -lblas -lboost_program_options
TEST_LIBS := -lblas -lboost_unit_test_framework -lstdc++fs

# Doxygen configuration file
DOXYFILE := Doxyfile

# Source files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
TEST_SRCS := $(wildcard $(TESTS_DIR)/*.cpp)
EXCLUDE_OBJ := $(BUILD_DIR)/LidDrivenCavitySolver.o

# Object files
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
TEST_OBJS := $(patsubst $(TESTS_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(TEST_SRCS))
NO_MAIN_OBJS := $(filter-out $(EXCLUDE_OBJ), $(OBJS))

# Main executable
EXEC := $(BIN_DIR)/solver

# Unit test executable
TEST_EXEC := $(BIN_DIR)/unittests

# Default target
default: $(EXEC)

# Compile main program
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

# Compile unit test files
$(BUILD_DIR)/%.o: $(TESTS_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(TEST_CXXFLAGS) -I$(INCLUDE_DIR) -o $@ -c $<

# Build main executable
$(EXEC): $(OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Build unit test executable
unittests: $(TEST_EXEC)
$(TEST_EXEC): $(TEST_OBJS) $(NO_MAIN_OBJS)
	$(CXX) $(TEST_CXXFLAGS) -o $@ $^ $(LIBS) $(TEST_LIBS)

# Clean
clean:
	-rm -rf $(BUILD_DIR)/*
	-rm -rf $(BIN_DIR)/*
	-rm -rf $(DOC_DIR)/*

# Generate documentation
docs:
	mkdir -p $(DOC_DIR)
	doxygen $(DOXYFILE)

# PHONY targets
.PHONY: default unittests clean doc

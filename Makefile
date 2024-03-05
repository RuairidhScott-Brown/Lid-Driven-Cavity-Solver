default: solver

CXX = g++
CC = g++
CXXFLAGS = -std=c++11 -Wall -O3
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
DOC_DIR = doc
TESTS_DIR = unittests
LDLIBS   = -lblas -lboost_program_options

build/SolverCG.o: src/SolverCG.cpp include/SolverCG.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDLIBS)

build/LidDrivenCavitySolver.o: src/LidDrivenCavitySolver.cpp include/LidDrivenCavity.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDLIBS)

build/LidDrivenCavity.o: src/LidDrivenCavity.cpp include/LidDrivenCavity.h include/SolverCG.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< $(LDLIBS)

solver: build/LidDrivenCavitySolver.o build/SolverCG.o build/LidDrivenCavity.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: clean

clean:
	-rm -rf build/*
	-rm solver
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(pybind11_relu LANGUAGES CXX)
add_subdirectory(pybind11)
include_directories(./include)
#include_directories(../../include)

pybind11_add_module(relu ./src/relu.cpp)

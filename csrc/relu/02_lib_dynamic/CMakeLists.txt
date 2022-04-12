cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(librelu LANGUAGES CXX)

add_library(relu SHARED main.cpp relu.h)

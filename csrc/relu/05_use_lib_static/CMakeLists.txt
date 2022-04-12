cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(main LANGUAGES CXX)
link_directories(../04_lib_static/build)

add_executable(main main.cpp relu.h)
target_link_libraries(main librelu.a)

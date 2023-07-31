set(TARGET_NAME "OwaranaiEngineMain")
project(${TARGET_NAME})

set(CMAKE_CXX_FLAGS "-g -Wall -O2")

cmake_minimum_required(VERSION 3.8)
add_subdirectory(Code)
add_executable(${TARGET_NAME} main.cpp)
target_link_libraries (${TARGET_NAME} OwaranaiEngine)

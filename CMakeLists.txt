set(TARGET_NAME "OwaranaiEngineMain")
project(${TARGET_NAME})

set(CMAKE_CXX_FLAGS "-g -Wall -O2")
set(CMAKE_CXX_STANDARD 17)

set(BACKWARD "NONE" CACHE STRING "Set the mode of operation")
set(CGTYPE "DYNAMIC" CACHE STRING "Set the mode of computational graph")

cmake_minimum_required(VERSION 3.8)
add_subdirectory(Code)
add_subdirectory(Application)
add_executable(${TARGET_NAME} main.cpp)
target_link_libraries (${TARGET_NAME} OwaranaiEngine)
target_link_libraries (${TARGET_NAME} Application)
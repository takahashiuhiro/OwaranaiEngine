project(OwaranaiEngineTest)

set(CMAKE_CXX_FLAGS "-g -Wall -O2")

cmake_minimum_required(VERSION 3.8)
add_subdirectory(OwaranaiEngine)
ADD_EXECUTABLE(OwaranaiEngineTest main.cpp)
target_link_libraries (OwaranaiEngineTest OwaranaiEngine)

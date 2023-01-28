project(OwaranaiEngineTest)

# required cmake version
cmake_minimum_required(VERSION 3.8)
find_package(CUDA)
if(CUDA_FOUND)
    add_definitions(-DCUDA_USEFUL)
else()
    add_definitions(-DCUDA_NOT_USEFUL)
endif()
add_subdirectory(OwaranaiEngine)
ADD_EXECUTABLE(OwaranaiEngineTest main.cpp)
target_link_libraries (OwaranaiEngineTest OwaranaiEngine)

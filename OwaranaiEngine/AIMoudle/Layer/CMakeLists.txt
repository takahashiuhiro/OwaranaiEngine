project(Layer)

# required cmake version
cmake_minimum_required(VERSION 3.8)


find_package(CUDA)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(Layer SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(Layer STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

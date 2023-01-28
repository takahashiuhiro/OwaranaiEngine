project(TensorCoreCPU)

# required cmake version
cmake_minimum_required(VERSION 3.8)


file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(TensorCore SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(TensorCoreCPU STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
project(GeometryMoudle)

# required cmake version
cmake_minimum_required(VERSION 3.8)


find_package(CUDA)

#add_subdirectory(TensorCore)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
cuda_add_library(GeometryMoudle STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

#target_link_libraries (AIMoudle TensorCore)
project(CommonMathMoudle)

# required cmake version
cmake_minimum_required(VERSION 3.8)


add_subdirectory(TensorCore)
add_subdirectory(GeometryCore)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(CommonMathMoudle STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries (CommonMathMoudle TensorCore)
target_link_libraries (CommonMathMoudle GeometryCore)
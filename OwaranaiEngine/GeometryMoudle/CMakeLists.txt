project(GeometryMoudle)

# required cmake version
cmake_minimum_required(VERSION 3.8)



file(GLOB_RECURSE CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.cpp )

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(GeometryMoudle STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

#target_link_libraries (AIMoudle TensorCore)
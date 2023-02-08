project(AIMoudle)

# required cmake version
cmake_minimum_required(VERSION 3.8)


#add_subdirectory(TensorCore)
add_subdirectory(Ops)
add_subdirectory(Optimizer)
add_subdirectory(Layer)
add_subdirectory(ComputationalGraph)
add_subdirectory(LossFunction)
add_subdirectory(Helpers)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(AIMoudle STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

#target_link_libraries (AIMoudle TensorCore)
target_link_libraries (AIMoudle Ops)
target_link_libraries (AIMoudle Optimizer)
target_link_libraries (AIMoudle Layer)
target_link_libraries (AIMoudle ComputationalGraph)
target_link_libraries (AIMoudle LossFunction)
target_link_libraries (AIMoudle Helpers)
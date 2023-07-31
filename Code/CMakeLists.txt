project(OwaranaiEngine)

cmake_minimum_required(VERSION 3.8)

add_subdirectory(CommonMathMoudle)
add_subdirectory(CommonDataStructure)
add_subdirectory(AutomaticDifferentiation)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
add_library(OwaranaiEngine STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries (OwaranaiEngine CommonMathMoudle)
target_link_libraries (OwaranaiEngine CommonDataStructure)
target_link_libraries (OwaranaiEngine AutomaticDifferentiation)
project(OwaranaiEngine)

cmake_minimum_required(VERSION 3.8)

add_subdirectory(AIMoudle)
add_subdirectory(CommonMathMoudle)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

#cuda_add_library(AIMoudle SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
find_package(Threads REQUIRED)
add_library(OwaranaiEngine STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries (OwaranaiEngine Threads::Threads)
target_link_libraries (OwaranaiEngine AIMoudle)
target_link_libraries (OwaranaiEngine CommonMathMoudle)
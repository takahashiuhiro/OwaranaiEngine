set(TARGET_NAME "CommonMathMoudleGPU")
project(${TARGET_NAME})

cmake_minimum_required(VERSION 3.8)

find_package(CUDA)
find_library(CURAND_LIBRARY curand)

file(GLOB_RECURSE CURRENT_HEADERS *.cuh)
file(GLOB_RECURSE CURRENT_SOURCES *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

cuda_add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_compile_definitions(${TARGET_NAME} PUBLIC USE_CUBLAS)

target_link_libraries(${TARGET_NAME} ${CURAND_LIBRARY})
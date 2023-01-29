#project(TensorCore)
set(TARGET_NAME "TensorCore")

# required cmake version
#cmake_minimum_required(VERSION 3.8)

find_package(CUDA)



if(CUDA_FOUND)
    add_subdirectory(GPUVer)
    file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
    file(GLOB CURRENT_SOURCES  *.cpp *.cu)
    source_group("Include" FILES ${CURRENT_HEADERS}) 
    source_group("Source" FILES ${CURRENT_SOURCES}) 
    cuda_add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
    target_link_libraries (${TARGET_NAME} TensorCoreGPU)
else()
    add_subdirectory(CPUVer)
    file(GLOB_RECURSE CURRENT_HEADERS  *.h)
    file(GLOB CURRENT_SOURCES  *.cpp)
    source_group("Include" FILES ${CURRENT_HEADERS}) 
    source_group("Source" FILES ${CURRENT_SOURCES}) 
    add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC CUDA_NOT_USEFUL)
    target_link_libraries (${TARGET_NAME} TensorCoreCPU)
endif()
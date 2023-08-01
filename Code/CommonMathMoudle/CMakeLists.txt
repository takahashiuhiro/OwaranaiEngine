set(TARGET_NAME "CommonMathMoudle")
project(${TARGET_NAME})

find_package(CUDA)

if(CUDA_FOUND)
    add_subdirectory(cuda)
endif()

file(GLOB CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.cpp)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES})

if(CUDA_FOUND)
    cuda_add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC CUDA_USEFUL)
    target_compile_definitions(${TARGET_NAME} PUBLIC THREAD_USEFUL)
    target_link_libraries (${TARGET_NAME} CommonMathMoudleGPU)
else()
    add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC CUDA_NOT_USEFUL)
    target_compile_definitions(${TARGET_NAME} PUBLIC THREAD_NOT_USEFUL)
endif()

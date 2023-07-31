set(TARGET_NAME "CommonDataStructure")
project(${TARGET_NAME})


file(GLOB_RECURSE CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.cpp)


source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES})


add_library(${TARGET_NAME} STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})



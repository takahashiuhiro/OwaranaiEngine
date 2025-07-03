set(TARGET_NAME "BlackBoxOptimizer")
project(${TARGET_NAME})

file(GLOB_RECURSE CURRENT_HEADERS  *.h)

add_library(${TARGET_NAME} INTERFACE)

source_group("Include" FILES ${CURRENT_HEADERS})

target_include_directories(${TARGET_NAME}
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)


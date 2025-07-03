project(OwaranaiEngine)

cmake_minimum_required(VERSION 3.8)

add_subdirectory(CommonMathMoudle)
add_subdirectory(CommonDataStructure)
add_subdirectory(DynamicAutomaticDifferentiation)
add_subdirectory(BlackBoxOptimizer)

add_library(OwaranaiEngine INTERFACE)

target_link_libraries(OwaranaiEngine
    INTERFACE
        CommonMathMoudle
        CommonDataStructure
        DynamicAutomaticDifferentiation
        BlackBoxOptimizer
)

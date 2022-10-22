project(OwaranaiEngineTest)

# required cmake version
cmake_minimum_required(VERSION 3.8)


add_subdirectory(OwaranaiEngine)
#set (EXTRA_LIBS ${EXTRA_LIBS} TensorCore)


cuda_ADD_EXECUTABLE(OwaranaiEngineTest main.cpp)


target_link_libraries (OwaranaiEngineTest OwaranaiEngine)

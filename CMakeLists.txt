project(OwaranaiEngine)

# required cmake version
cmake_minimum_required(VERSION 3.8)


add_subdirectory(AIMoudle)
#set (EXTRA_LIBS ${EXTRA_LIBS} TensorCore)


cuda_ADD_EXECUTABLE(OwaranaiEngine main.cpp)


target_link_libraries (OwaranaiEngine AIMoudle)

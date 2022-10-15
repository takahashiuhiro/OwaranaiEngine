project(OwaranaiEngine)

# required cmake version
cmake_minimum_required(VERSION 3.8)


add_subdirectory(TensorCore)
#set (EXTRA_LIBS ${EXTRA_LIBS} TensorCore)


ADD_EXECUTABLE(OwaranaiEngine main.cpp)


target_link_libraries (OwaranaiEngine TensorCore)

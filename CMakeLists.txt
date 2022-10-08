


# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.16)


# 项目信息
project(OwaranaiEngine)




# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加子目录
add_subdirectory(lib/TensorCore)
# 指定生成目标
add_executable(OwaranaiEngine main.cpp)
# 添加链接库
target_link_libraries(OwaranaiEngine TensorCore)
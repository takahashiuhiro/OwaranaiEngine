#include <memory>
#include <cmath>
#include <fstream>
#include <functional>
#include <stack>
#include <iostream>
#include "Code/OEDynamic.h"

// 变长参数模板函数
template <typename... Args>
void myFunction(Args... args) {
    (std::cout << ... << args) << std::endl;  // 使用折叠表达式打印所有参数
}

// 定义一个函数，接受变长参数模板函数指针作为参数
template <typename... Args>
void callVarFunc(void (*func)(Args...), Args... args) {
    func(args...);
}

int main() {
    callVarFunc(myFunction<int, double, char, const char*>, 1, 2.5, 'a', "hello");
    return 0;
}
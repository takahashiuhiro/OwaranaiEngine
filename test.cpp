#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include "Code/CommonMathMoudle/OpenGL/GLSL.h"



void checkShaderCompilation(GLuint shader) {
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkProgramLinking(GLuint program) {
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking failed: " << infoLog << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // 初始化 GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // 设置 OpenGL 版本（4.3 或更高版本以支持计算着色器）
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // 隐藏窗口

    // 创建一个隐藏的窗口以创建 OpenGL 上下文
    GLFWwindow* window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // 初始化 GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // 定义矩阵大小
    const int matrixSize = 16; // 4x4 矩阵
    std::vector<float> A(matrixSize);
    std::vector<float> B(matrixSize);
    std::vector<float> C(matrixSize, 0.0f);

    // 初始化矩阵 A 和 B
    for (int i = 0; i < matrixSize; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(matrixSize - i);
    }

    auto gg = GLSL::I().GetFunStr(GLSL::I().AddInCPP);

    // 创建并编译计算着色器
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &gg, nullptr);
    glCompileShader(computeShader);
    checkShaderCompilation(computeShader);

    // 创建程序并链接
    GLuint program = glCreateProgram();
    glAttachShader(program, computeShader);
    glLinkProgram(program);
    checkProgramLinking(program);

    // 删除着色器对象
    glDeleteShader(computeShader);

    // 创建缓冲区对象并上传数据
    GLuint buffers[5];
    glGenBuffers(5, buffers);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * sizeof(float), nullptr, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * sizeof(float), A.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);

    auto asize = A.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(size_t), &asize, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, matrixSize * sizeof(float), B.data(), GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers[3]);

    auto bsize = B.size();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(size_t), &bsize, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers[4]);

    // 使用程序并执行计算着色器
    glUseProgram(program);
    glDispatchCompute(1, 1, 1);//这个可以设置工作组，和cuda一样的
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);//类似于__syncthreads但是是全局的，在glsl里的话可以用barrier();

    // 读取结果
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);//执行了这一句后面的操作就会绑定到这里
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, matrixSize * sizeof(float), C.data());

    // 输出结果
    std::cout << "Matrix C (Result of A + B):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << C[i * 4 + j] << " ";
        }
        std::cout << std::endl;
    }

    // 清理资源
    glDeleteBuffers(3, buffers);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

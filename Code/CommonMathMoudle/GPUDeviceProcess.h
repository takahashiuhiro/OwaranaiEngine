#include <iostream>
#include <vector>
#include "../CommonDataStructure/CommonFuncHelpers.h"
#include "../CommonDataStructure/Log.h"
#include "OpenGL/GLSL.h"

#define THREAD_NUM 256

#ifdef OPENGL_USEFUL
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#endif

#ifdef OPENGL_USEFUL
static GLuint GetBuffer_OpenGL_Inline(size_t ShapeCount, size_t TypeSize = sizeof(float))
{
    GLuint ResBuffer;
    glGenBuffers(1, &ResBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ResBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, ShapeCount * TypeSize, nullptr, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    return ResBuffer;
}

template<typename T>
static void DataCPUToGPU_Inline(GLuint InputBuffer, T*CPUDevicePointer, size_t ShapeCount)
{
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, InputBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, ShapeCount*sizeof(T), CPUDevicePointer, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}
#endif

struct VBuffer
{
    #ifdef OPENGL_USEFUL
    GLuint OpenGLTMPBuffer;
    #endif

    VBuffer(){};

    template<typename T>
    static VBuffer CVBuffer(T InputData)
    {
        VBuffer Res;
        Res.Init<T>(&InputData, 1);
        return Res;
    }

    template<typename T>
    static VBuffer CVBuffer(T* InputData ,size_t ShapeCount)
    {
        VBuffer Res;
        Res.Init<T>(InputData, ShapeCount);
        return Res;
    }

    template<typename T>
    void Init(T* InputData ,size_t ShapeCount)
    {
        #ifdef OPENGL_USEFUL
        OpenGLTMPBuffer = GetBuffer_OpenGL_Inline(ShapeCount);
        DataCPUToGPU_Inline(OpenGLTMPBuffer, InputData, ShapeCount);
        #endif
    }

    ~VBuffer()
    {
        #ifdef OPENGL_USEFUL
        glDeleteBuffers(1, &OpenGLTMPBuffer);
        #endif
    }
};


class GPUDeviceProcess {
private:
    GPUDeviceProcess() 
    {
        #ifdef OPENGL_USEFUL
        Init_OpenGL();
        CompileAllGPUFunction();
        #endif
    }
    ~GPUDeviceProcess()
    {
        #ifdef OPENGL_USEFUL
        /**
         * 删除编译过的程序
         */
        auto DeleteOpenGLPrograme = []()
        {
            for(auto&it:GPUDeviceProcess::I().GPUFunction_OpenGL)
            {
                glDeleteProgram(it);
            }
        };
        DeleteOpenGLPrograme();
        glfwDestroyWindow(window);
        glfwTerminate();
        #endif
    }

public:
    GPUDeviceProcess(const GPUDeviceProcess&) = delete;
    GPUDeviceProcess& operator=(const GPUDeviceProcess&) = delete;

    static GPUDeviceProcess& I() 
    {
        static GPUDeviceProcess instance; // 静态局部变量，线程安全
        return instance;
    }

    std::pair<size_t, size_t>GetWorkItem(size_t InputThreadNum)
    {
        return std::make_pair((InputThreadNum + THREAD_NUM - 1) / THREAD_NUM, THREAD_NUM);
    }

    #ifdef OPENGL_USEFUL

    GLFWwindow* window;
    std::vector<GLuint>GPUFunction_OpenGL;//储存OpenGL函数的容器
    
    /**
     * 对于opengl的计算后端进行初始化
     */
    void Init_OpenGL()
    {
        Log::Assert(glfwInit(), "Failed to initialize GLFW");
        // 设置 OpenGL 版本（4.3 或更高版本以支持计算着色器）
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE); // 隐藏窗口

        window = glfwCreateWindow(1, 1, "", nullptr, nullptr);
        Log::Assert(window, "Failed to create GLFW window");

        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        Log::Assert(glewInit() == GLEW_OK,"Failed to initialize GLEW");
    }

    /**.
     * 生成并返回一个buffer,解除绑定,这个时候数据已经再opengl管理的缓冲区内了
     */
    GLuint GetBuffer_OpenGL(size_t ShapeCount)
    {
        return GetBuffer_OpenGL_Inline(ShapeCount);
    }

    /**
     * 数据从GPU传入CPU
     */
    void DataGPUToCPU(GLuint InputBuffer, float*CPUDevicePointer, size_t ShapeCount)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, InputBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, ShapeCount * sizeof(float), CPUDevicePointer);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    /**
     * 数据从CPU传入GPU
     */
    void DataCPUToGPU(GLuint InputBuffer, float*CPUDevicePointer, size_t ShapeCount)
    {
        DataCPUToGPU_Inline(InputBuffer, CPUDevicePointer, ShapeCount);
    }

    void CheckShaderCompilation(GLuint shader) 
    {
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation failed: " << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void CheckProgramLinking(GLuint program) 
    {
        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::cerr << "Program linking failed: " << infoLog << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    /**
     * 编译OpenGL所有的GPU程序
     */
    void CompileAllGPUFunction()
    {
        for(int FunName = 0;FunName < GLSL::I().GLSLFunNum;FunName++)
        {
            auto GLSLFunctionStr = GLSL::I().GetFunStr(FunName);
            // 创建并编译计算着色器
            GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
            glShaderSource(computeShader, 1, &GLSLFunctionStr, nullptr);
            glCompileShader(computeShader);
            CheckShaderCompilation(computeShader);
            // 创建程序并链接
            GLuint program = glCreateProgram();
            glAttachShader(program, computeShader);
            glLinkProgram(program);
            CheckProgramLinking(program);
            // 删除着色器对象
            glDeleteShader(computeShader);
            GPUFunction_OpenGL.push_back(program);
        }
    }

    void BindVariable(GLint InputBuffer, int BindIndex)
    {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, InputBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BindIndex, InputBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }

    void BindVariableVec(std::vector<GLint>InputVariables)
    {
        for(int a=0;a<InputVariables.size();a++)
        {
            BindVariable(InputVariables[a],a);
        }
    }

    void UnBindVariable(int BindNum)
    {
        for(int a=0;a<BindNum;a++)
        {
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, a, 0);
        }
    }

    GLint GetGPUFunction(int FuntionIndex)
    {
        return GPUFunction_OpenGL[FuntionIndex];
    }


    void ProcessGLSLFun(int GLSLFunName ,int WorkNum ,std::vector<GLint> FunctionParams)
    {
        BindVariableVec(FunctionParams);
        glUseProgram(GetGPUFunction(GLSLFunName));
        glDispatchCompute(GetWorkItem(WorkNum).first, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        UnBindVariable(FunctionParams.size());
    }

    #endif
};

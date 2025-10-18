# README
## 关于项目内容介绍  

项目内容为无Cuda/OpenGL外其他依赖的纯净C++计算框架, 目前其中主要包含几个模块:  

### 深度学习引擎相关
1. 自动微分  
2. 张量的基本操作   
3. 神经网络层(类似nn.module, 对计算图进行封装, 以达到方便使用, 复用逻辑的目的)   
4. 优化器(更新神经网络)  
5. Loss  

### 黑箱优化相关
1. 基于NES对优化目标的score function最大化的参数求解  

### 应用相关
1. 复现了基于OwaranaiEngine的nanoGPT，可简单使用   
## 测试

### GPT2 拟合测试
训练过程可见: https://github.com/takahashiuhiro/OwaranaiEngine/blob/main/Test/GPTX/test_res/test.md
模型可以正常拟合数据集并在训练集上输出有一定逻辑的结果，loss曲线也和基于torch的GPT2相似，OwaranaiEngine在该模型下被验证可用。

测试过程中使用OwaranaiEngine重写了基于torch的gpt参考了该链接的代码，并且在单元测试中有很大的帮助。在这里需要感谢
https://github.com/karpathy/nanoGPT/tree/master

## 项目文档  

todo

## 项目运行方法  

### Common
git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  

### CPU ONLY
1. 编译运行
g++ --std=c++17  -DOPENGL_USEFUL ../main.cpp -o main  
./main
2. 使用C++解释器 Cling直接运行
Cling --std=c++17 ../main.cpp

### OPENGL
g++  -DOPENGL_USEFUL ../main.cpp -o main  -lGL -lGLEW -lglfw
./main

### CUDA
nvcc -std=c++17 -c ../Cuda/TensorCoreGPU.cu  -o tensorcuda.o
g++ -std=c++17 -DCUDA_USEFUL ../main.cpp tensorcuda.o  -o main  -L/usr/local/cuda/lib64 -lcurand -lcudart -O2
./main

## 项目备忘录 
1. hyperelement的splay还没写 
2. 多元高斯分布的cuda版没写 
3. SVD分解没写  
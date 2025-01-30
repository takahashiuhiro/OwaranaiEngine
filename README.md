# README
## 关于项目内容介绍  

项目内容为纯C++手写深度学习框架项目, 目前其中主要包含几个模块:  

1. 对于深度学习框架的核心自动微分模块提供了两种不同的实现，动态计算图/静态计算图  
2. 张量的基本操作  
3. 计算图与自动微分(前向与反向)  
4. 神经网络层(类似nn.module, 对计算图进行封装, 以达到方便使用, 复用逻辑的目的)  
5. 优化器(更新神经网络)  
6. Loss  
7. 当然，也能找到的一个奇怪的自制容器(HyperElement)，是为了更好存取参数使用的 

在计算图中采用了显式的图结构, 可以打印出计算图的邻接表以及每个计算节点的梯度. 如果采用的算子有多阶导数就可以求多阶导数。

## 关于本框架测试

### GPT2 拟合测试
训练过程可见: https://github.com/takahashiuhiro/OwaranaiEngine/blob/main/Application/GPTX/test_res/test.md
模型可以正常拟合数据集并在训练集上输出有一定逻辑的结果，loss曲线也和基于torch的GPT2相似，OwaranaiEngine在该模型下被验证可用。

测试过程中使用OwaranaiEngine重写了基于torch的gpt参考了该链接的代码，并且在单元测试中有很大的帮助。在这里需要感谢
https://github.com/karpathy/nanoGPT/tree/master

## 关于项目文档  

https://github.com/takahashiuhiro/OwaranaiEngine/blob/main/doc.md 

## 关于本项目的配置  

(必须)C++ 11  
(选装)Cuda Version:11.8  
(选装)Cuda Drive Version:522.25  

只要装了g++就能跑起来, 检测不到cuda会编译C++版

## 关于项目运行方法  
1. wsl/ubuntu:  
git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  
cmake ../  
make  
./OwaranaiEngineMain  

2. windows:  
git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
新建一个空白的vs工程  
把整个项目根目录放入空白工程下  
打开工程  
在右侧资源管理器下找到OwaranaiEngine  
右键选中 包括在项目中  
右键解决方案 生成  
ctrl+F5编译运行  

## 项目备忘录 
1. hyperelement的splay还没写
2. DynamicTensor::Cat有严重的复杂度的性能问题，要重新实现，现在的问题在于cat的时候会将高维矩阵压平，但是亚平后可能会产生一个元素数*元素数级别的大稀疏矩阵，内存放不下。
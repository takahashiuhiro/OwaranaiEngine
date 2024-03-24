# README
## 关于项目内容介绍  

项目内容为纯C++手写深度学习框架项目, 目前其中主要包含几个模块:  
1. 张量的基本操作  
2. 计算图与自动微分(前向与反向)  
3. 神经网络层(类似nn.module, 对计算图进行封装, 以达到方便使用, 复用逻辑的目的)  
4. 优化器(更新神经网络)  
5. Loss  
6. 当然，也能找到的一个奇怪的自制容器(HyperElement)，是为了更好存取参数使用的 

在计算图中采用了显式的图结构, 可以打印出计算图的邻接表以及每个计算节点的梯度. 如果采用的算子有多阶导数就可以求多阶导数。

## 关于项目文档  

https://github.com/takahashiuhiro/OwaranaiEngine/blob/main/doc.md 

## 关于项目完成度和目标  

transformer施工中..

## 关于快速上手  

todo

## 关于本项目的配置  

(必须)C++ 11  
(选装)Cuda Version:11.8  
(选装)Cuda Drive Version:522.25  

只要装了g++就能跑起来, 检测不到cuda会编译C++版

## 关于项目运行方法  

git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  
cmake ../  
make  
./OwaranaiEngineMain  


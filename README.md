# README
## 关于项目内容介绍  

项目内容为纯C++手写深度学习框架项目, 目前其中主要包含几个模块:  
1. 张量的基本操作  
2. 计算图与自动微分(前向与反向)  
3. 神经网络层(类似nn.module, 对计算图进行封装, 以达到方便使用, 复用逻辑的目的)  
4. 优化器(更新神经网络)  
5. Loss  

在计算图中, 作者采用了显式的图结构, 可以打印出计算图的邻接表以及每个计算节点的梯度. 如果采用的算子有多阶导数, 那确实是有api可以一键求多阶导数的.

PS: 作者并不太会GPU编程, 因此该部分的运行效率应该是不堪入目, 只能说是勉强能跑? 等日后项目基本完成后可能会回来优化或者重写, 但是现在并不会做这件事.  
项目为作者单人进行制作的独立项目, 对于深度学习, C++和数学的了解都非常浅显, 因此本项目实际运行起来效率应该不会很高, 麻雀虽小但五脏俱全就是作者本人的目标.  

## 关于项目文档  

详细的文档位于根目录的doc.md  

## 关于项目完成度和目标  

项目的release 0.1目标为完成一个完整的transformer, 因此项目的进行会随着transformer需要的方向进行搭建, 会先写transformer所需要的部分.  

## 关于快速上手  

可以看TestSample\LinearLayerTest.cpp, 这个文件是一个y = kx的基本拟合用法, 复制内容到main.cpp即可运行。
文档部分还没开始动手, 但是注释写的自认为还是比较用心的.(可能存在一些工地英语, 作者语言水平拉闸求别笑..

## 关于本项目的配置  

Cuda Version:11.8  
Cuda Drive Version:522.25  
C++ 11  

只要装了g++就能跑起来,无所谓是否有cuda,如果机器里没有英伟达显卡会编成CPU版. 

## 关于项目运行方法  

git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  
cmake ../  
make  
./OwaranaiEngineMain  


## TODO  
 1.GPT2  
    1.nn.GELU  
    2.nn.Linear  
    3.nn.Embedding  
 2.文档  
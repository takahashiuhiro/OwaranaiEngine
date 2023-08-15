# README
## 本项目的配置  

Cuda Version:11.8  
Cuda Drive Version:522.25  
C++ 11  
理论上来说，只要装了nvcc/g++/cuda驱动版本合理就应该能跑起来  

##  项目运行方法  

git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  
cmake ../  
make  
./OwaranaiEngineTest  

## 项目目录  

OwaranaiEngine  项目根目录  
----Code  代码  
--------AutomaticDifferentiation  自动微分  
------------Ops  算子  
------------其他  计算图相关  
--------CommonDataStructure  通用数据结构  
--------CommonMathMoudle  通用数学  
------------Cuda  一些cuda的基础操作实现  
------------其他  张量相关  
----TestSample  测试模块用的各种main文件  


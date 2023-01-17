# README
## 本项目的配置  

Cuda Version:11.8  
Cuda Drive Version:522.25  
C++ 11  
理论上来说，只要装了nvcc/g++/cuda驱动版本合理就应该能跑起来  

--------------------------
##  项目运行方法  

cd build  
cmake ../  
make  
./OwaranaiEngineTest  
执行完这些不出意外就跑的起来  

----------------------------
## 项目目录  

OwaranaiEngine  项目根目录  
        AIMoudle  AI模块  
                ComputationalGraph 计算图     
                Layer 网络层  
                Ops  算子  
                Optimizer  优化器  
                TensorCore  张量计算  
                         Cuda  一些cuda的基础操作实现  
                Helpers  杂项类  
                LossFunction  用于写loss函数的地方  
        GeometryMoudle  几何模块  

----------------------------
## 最近的一些计划

1.跑起来一个简单的手写数字识别  
2.搭建一些前向算法基础 


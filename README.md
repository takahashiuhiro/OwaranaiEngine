# README

本项目的配置采用了   

Cuda Version:11.8  

Cuda Drive Version:522.25  

C++ 11  

理论上来说，只要装了nvcc/g++/cuda驱动版本合理就应该能跑起来....   

--------------------------

cd build  

cmake ../  

make  

./OwaranaiEngineTest  

执行完这些不出意外就跑的起来

----------------------------



本项目想达成的目标同库的名字一样，是不会结束的，短期目标是实现一个可以对传统路径规划算法进行数值优化的框架  

## 虽然想用英语写注释，但是我发现了我的英语水平真的不行

项目目录:  

OwaranaiEngine  项目根目录  

    AIMoudle  AI模块  

        ComputationalGraph 计算图     

        Layer 网络层  

        Ops  算子  

        Optimizer  优化器  

        TensorCore  张量计算  

            Cuda  一些cuda的基础操作实现  

    GeometryMoudle  几何模块  
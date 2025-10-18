# OwaranaiEngine  

A lightweight, pure **C++ header-only computational framework** focused on automatic differentiation and numerical computing.  
It has **no external dependencies** other than optional **CUDA** and **OpenGL** backends, and can run seamlessly on **CPU / OpenGL / CUDA** environments.  

---

## Overview  

**OwaranaiEngine** is a research-oriented framework for experimental development of automatic differentiation, tensor computation, and information-geometric black-box optimization — all written in modern C++.  

The framework is entirely **header-only**, requiring no additional build steps beyond `#include`.  
It also supports **interactive C++ interpreters** (such as **Cling**) for rapid prototyping and experimentation.  

---

## Core Modules  

### Deep Learning Engine  
1. **Automatic Differentiation** – dynamic computational graph with backpropagation  
2. **Tensor Operations** – broadcasting, matrix computation, and element-wise operations  
3. **Neural Network Layers** – an `nn.Module`-like abstraction for reusable computation graphs  
4. **Optimizers** – gradient-based parameter update mechanisms  
5. **Loss Functions** – standard loss components for supervised learning  

---

### Black-Box Optimization  
1. **NES-based parameter estimation** that maximizes the score function associated with the optimization objective.

---

### Applications  
- Reimplementation of **nanoGPT** using OwaranaiEngine (a simplified GPT-2 model)  
- The model successfully converges on training data with loss curves similar to PyTorch GPT-2  
- Training logs and results:  
  [Test/GPTX/test_res/test.md](https://github.com/takahashiuhiro/OwaranaiEngine/blob/main/Test/GPTX/test_res/test.md)  

Special thanks to [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) for the original reference implementation.  

---

## Build & Run  

### 🔹 Common Setup  
git clone https://github.com/takahashiuhiro/OwaranaiEngine.git  
cd build  

---

###  CPU Only  
#### Compile & Run  
g++ -std=c++17 ../main.cpp -o main  
./main  

#### Or run directly with Cling  
cling --std=c++17 ../main.cpp  

---

###  OpenGL Backend  
g++ -std=c++17 -DOPENGL_USEFUL ../main.cpp -o main -lGL -lGLEW -lglfw  
./main  

---

###  CUDA Backend  
nvcc -std=c++17 -c ../Cuda/TensorCoreGPU.cu -o tensorcuda.o  
g++ -std=c++17 -DCUDA_USEFUL ../main.cpp tensorcuda.o -o main -L/usr/local/cuda/lib64 -lcurand -lcudart -O2  
./main  

---

##  Documentation  

**TODO**  

- `Hyperelement` splay structure not yet implemented  
- CUDA version of multivariate Gaussian distribution pending  
- SVD decomposition not yet implemented  


##  Acknowledgments  
Special thanks to **[Andrej Karpathy](https://github.com/karpathy)** for the open-source project **nanoGPT**,  
which served as a valuable reference during the GPT-2 testing phase of OwaranaiEngine.  

---

##  License  
MIT License © TakahashiUhiro  

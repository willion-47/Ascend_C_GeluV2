# GeluV2 Custom Operator for CANN

[![CANN Version](https://img.shields.io/badge/CANN-7.0%2B-blue.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-Orange%20Pi%20AI%20Pro-orange.svg)]()
[![Language](https://img.shields.io/badge/Language-Ascend%20C-green.svg)]()

## 📖 简介 (Introduction)

本项目提供了一个基于华为 CANN 异构计算架构开发的 **GeluV2** 自定义算子。该算子使用 Ascend C 编写，并针对搭载昇腾 AI 处理器的硬件设备（如 Orange Pi AI Pro）进行了深度性能优化。

GELU (Gaussian Error Linear Unit) 是一种广泛应用于 Transformer 架构（如 BERT, GPT 系列大模型）的激活函数。本算子严格实现了其数学核心逻辑：
`GELU(x) = x * Phi(x)`
其中，`Phi(x)` 为标准正态分布的累积分布函数。

$$\text{GELU}(x) = x \cdot \Phi(x)$$

$$\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} \exp\left(-\frac{t^2}{2}\right) dt$$

## ⚙️ 环境要求 (Prerequisites)

- **硬件:** Orange Pi AI Pro (昇腾 310B 系列芯片) / 其他 Ascend NPU
- **操作系统:** Ubuntu 22.04 / openEuler (aarch64 架构)
- **依赖软件:**
  - CANN Toolkit
  - CMake >= 3.16
  - gcc/g++ 交叉编译工具链 (针对 aarch64)

## 📐 算子规格 (Operator Specification)

| 算子类型 (OpType) | `GeluV2` |
| :--- | :--- |
| **输入 `x`** | 数据类型: `float16`, `float32` <br> 格式: `ND` |
| **输出 `y`** | 数据类型: `float16`, `float32` (与输入保持一致) <br> 格式: `ND` |

## 📂 核心代码结构 (Directory Structure)

```text
├── op_host
│   ├── CMakeLists.txt         # Host侧编译配置
│   ├── gelu_v2.cpp            # Host侧算子原型注册与多核 Tiling 切分逻辑
│   └── gelu_v2_tiling.h       # Tiling 数据结构定义
├── op_kernel
│   ├── CMakeLists.txt         # Kernel侧编译配置
│   └── gelu_v2.cpp            # Device侧 Ascend C Kernel 计算逻辑实现
└── custom_opp_ubuntu_aarch64.run # 编译生成的算子部署包

```

## 🚀 部署与安装 (Installation)

本项目已提供编译完成的算子 `.run` 包，可直接在目标硬件（如 Orange Pi AI Pro）上执行安装。

1. **设置 CANN 环境变量**:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

```


2. **赋予执行权限并安装**:
```bash
chmod +x custom_opp_ubuntu_aarch64.run
./custom_opp_ubuntu_aarch64.run

```


*安装成功后，算子将自动注册到 CANN 的算子库中 (通常位于 `~/.local/Ascend/opp` 或系统级的 `/usr/local/Ascend/opp` 路径下)。*

## 🧠 Tiling 与性能优化 (Performance Optimization)

在 `op_host/gelu_v2.cpp` 中，算子实现了动态多核 Tiling 策略：

* **Block切分**: 根据输入张量的 `totalLength` 与硬件获取的 `coreNum`，动态计算每个 AI Core 的 `blockLength`。
* **内存对齐**: 严格处理 32-Byte 数据对齐，优化 NPU Vector 单元的访存带宽。
* **尾核处理**: 针对无法整除的数据量，精准计算 `lastBlockLength`，确保边界数据计算的正确性，不越界、不遗漏。

## 📄 许可证 (License)

本项目采用 Apache License 2.0 许可证。


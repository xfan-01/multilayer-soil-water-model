# Multi-layer Soil Water Model Based on Bidirectional Percolation
# 基于双向渗透的多层土壤水分模型

## Language / 语言选择

- **[English Version](./docs/README_EN.md)** - Complete documentation in English
- **[中文版本](./docs/README_CN.md)** - 完整的中文文档

---

## Quick Overview / 项目概述

This project implements a physics-based multi-layer soil water dynamics simulation model with natural bidirectional percolation flow mechanisms.

本项目实现了基于物理机制的多层土壤水分动态模拟模型，采用自然双向渗透流动机制。

### Key Features / 主要特性

- **Physical Consistency** / **物理一致性**: Natural water flow from wet to dry areas / 水从湿区→干区的自然流动
- **Bidirectional Percolation** / **双向渗透**: Based on saturated/unsaturated percolation mechanisms / 基于饱和/非饱和渗透机制
- **Multi-layer Structure** / **多层结构**: Supports arbitrary soil profile layers / 支持任意层数土壤剖面
- **KGE Optimization** / **KGE优化**: Parameter calibration using Kling-Gupta Efficiency / 使用KGE进行参数校准
- **Data-driven** / **数据驱动**: Based on Hyytiälä Forest Research Station data / 基于芬兰Hyytiälä研究站数据

---

## Quick Start / 快速开始

```bash
# Navigate to core directory / 进入核心代码目录
cd core

# Run the model / 运行模型
python model_driver.py
```

Or use Python import / 或使用Python导入:

```python
import sys
sys.path.append('./core')
from model_driver import main

# Run complete model workflow / 运行完整模型工作流
main()
```

For detailed documentation, please select your preferred language above.  
详细文档请选择上方您偏好的语言版本。

---

## Configuration Examples / 配置示例

### Model Configuration / 模型配置 (`config/model_config.txt`)

```ini
# Basic Model Configuration / 基本模型配置
n_layers = 3
max_iterations = 100
convergence_tolerance = 1e-6

# Physical Parameters / 物理参数
default_alpha = 0.1
default_beta = 0.05
default_max_theta_base = 45

# Visualization / 可视化配置
figure_dpi = 300
layer_colors = blue,green,red
font_family = Arial
```

### Variable Mapping / 变量映射 (`config/variable_mapping.txt`)

```ini
# Data Files / 数据文件
train_data_file = 2000.xlsx
test_data_file = 2001.xlsx

# Meteorological Variables / 气象变量
precipitation = P
evapotranspiration = E
time_column = Time

# Soil Layers / 土壤层配置
theta_10mm = theta10
theta_20mm = theta20
theta_50mm = theta50  # 3-layer mode enabled / 3层模式已启用
```

### Custom Configuration Example / 自定义配置示例

```python
import sys
sys.path.append('./core')
from config_manager import ConfigManager
from model_driver import train_and_evaluate_model

# Load custom configuration / 加载自定义配置
config = ConfigManager('../config/custom_config.txt', '../config/custom_mapping.txt')

# Run model with custom settings / 使用自定义设置运行模型
results = train_and_evaluate_model(config)
```

---

## File Structure / 文件结构

```
multilayer-soil-water-model/
├── core/                   # Core code files / 核心代码文件
│   ├── model_driver.py     # Main program driver / 主程序驱动器
│   ├── soil_model.py       # Core model algorithms / 土壤模型核心算法
│   ├── config_manager.py   # Configuration management / 配置管理模块
│   └── visualization.py    # Visualization module / 可视化模块
├── config/                 # Configuration files / 配置文件
│   ├── model_config.txt    # Model parameters / 模型参数配置
│   └── variable_mapping.txt # Variable mapping / 变量映射配置
├── data/                   # Data files / 数据文件
│   ├── 2000.xlsx          # Training data (2000) / 训练数据（2000年）
│   └── 2001.xlsx          # Test data (2001) / 测试数据（2001年）
├── docs/                   # Documentation / 文档文件
│   ├── README_CN.md       # Chinese documentation / 中文详细文档
│   └── README_EN.md       # English documentation / 英文详细文档
├── results/                # Output results / 结果文件
│   ├── model_performance_metrics.json  # Performance metrics / 性能指标数据
│   ├── train_comparison_3layer.png     # Training comparison / 训练对比图
│   ├── test_comparison_3layer.png      # Test comparison / 测试对比图
│   └── scatter_plots_3layer.png        # Scatter plots / 散点图分析
└── README.md              # Main documentation / 主文档
```

---

## Simulation Results / 模拟结果

### 3-Layer Model Performance / 3层模型表现

Latest simulation results for 3-layer soil moisture model:

**Training Performance (2000) / 训练表现:**
- **10cm layer**: NSE=0.37, PBIAS=-2.8% (Moderate Confidence / 中等可信度)
- **20cm layer**: NSE=0.19, PBIAS=-15.9% (Low Confidence / 低可信度)  
- **50cm layer**: NSE=-6.1, PBIAS=62.9% (Low Confidence / 低可信度)

**Test Performance (2001) / 测试表现:**
- **10cm layer**: NSE=-1.28, PBIAS=7.7% (Moderate Confidence / 中等可信度)
- **20cm layer**: NSE=-0.79, PBIAS=-3.1% (Moderate Confidence / 中等可信度)
- **50cm layer**: NSE=-11.6, PBIAS=66.2% (Low Confidence / 低可信度)

### Model Visualization Results / 模型可视化结果

#### Training Data Comparison / 训练数据对比
![Training Comparison](./results/train_comparison_3layer.png)

#### Test Data Comparison / 测试数据对比  
![Test Comparison](./results/test_comparison_3layer.png)

#### Scatter Plot Analysis / 散点图分析
![Scatter Plots](./results/scatter_plots_3layer.png)

### Generated Outputs / 生成的输出文件

- `results/model_performance_metrics.json` - Detailed performance metrics / 详细性能指标
- `results/train_comparison_3layer.png` - Training data comparison plots / 训练数据对比图
- `results/test_comparison_3layer.png` - Test data comparison plots / 测试数据对比图  
- `results/scatter_plots_3layer.png` - Scatter plot analysis / 散点图分析

---

## License / 许可证

This project is developed for academic research purposes.  
本项目用于学术研究目的开发。

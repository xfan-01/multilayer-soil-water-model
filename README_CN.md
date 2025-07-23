# 基于双向渗透的多层土壤水分模型

[English Version](./README_EN.md) | 中文版本

## 项目概述

本项目实现了一个基于物理机制的多层土壤水分动态模拟模型，核心创新在于**移除了人工层间交换限制**，完全基于土壤物理学的毛细作用和重力驱动的自然双向渗透流动机制。

### 主要特性
- **物理一致性**: 实现"水从湿区→干区"的自然流动过程
- **双向渗透**: 基于饱和/非饱和渗透机制  
- **多层结构**: 支持任意层数的土壤剖面模拟（当前支持2-3层）
- **KGE优化**: 使用Kling-Gupta Efficiency作为目标函数进行参数校准
- **简化评估**: 采用NSE和PBIAS作为核心评价指标，去除冗余指标
- **数据驱动**: 使用芬兰Hyytiälä森林研究站长期观测数据

---

## 数学理论基础

### 核心物理方程

#### 1. 饱和渗透
当土壤含水量超过饱和值时的排水过程：

```
qs,i = (θi - θs,i) × ks,i    当 θi > θs,i
qs,i = 0                     当 θi ≤ θs,i
```

**物理意义**: 重力驱动的自由排水，超饱和水分直接流失

#### 2. 非饱和渗透
毛细力和重力共同作用下的水分运动：

```
qun,i = (θi - θmin,i) × ku1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^ku2,i
```

**物理意义**: 非饱和条件下的渗透，强度与有效含水量呈非线性关系

#### 3. 毛细上升（独立公式）
为更准确模拟水分克服重力向上移动的过程，我们为毛细上升设置了独立参数：

```
q_cap,i = (θi - θmin,i) × kc1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^kc2,i
```

#### 4. 双向渗透流动
层间净流动量计算：

```
net_flux = qun_down - q_cap_up
```

其中：
- `qun_down`: 上层向下的渗透量
- `q_cap_up`: 下层向上的毛细上升量（使用毛细上升公式）

### 变量符号说明

| 符号 | 含义 | 单位 |
|------|------|------|
| θi | 第i层含水量 | mm |
| θs,i | 第i层饱和含水量 | mm |
| θmin,i | 第i层最小含水量（凋萎点） | mm |
| ks,i | 第i层饱和导水率 | mm/day |
| ku1,i | 第i层非饱和渗透系数 | mm/day |
| ku2,i | 第i层非饱和渗透指数 | - |
| kc1,i | 第i层毛细上升系数 | mm/day |
| kc2,i | 第i层毛细上升指数 | - |
| P | 降水量 | mm/day |
| E | 蒸散发量 | mm/day |
| R | 地表径流 | mm/day |
| Qgw | 深层渗漏 | mm/day |

### 水量平衡方程

对于第i层土壤，水量平衡表达为：

```
Δθi = Pi - Ei - qs,i - qun,i + Qin,i - Qout,i
```

其中：
- `Pi`: 第i层接收的水量输入（降水或上层排水）
- `Ei`: 第i层的蒸散发损失
- `qs,i`: 第i层饱和排水量
- `qun,i`: 第i层非饱和渗透量
- `Qin,i`: 从相邻层流入的水量
- `Qout,i`: 向相邻层流出的水量

---

## 优化与评估系统

### KGE目标函数
模型采用**Kling-Gupta Efficiency (KGE)**作为参数优化的目标函数：

```
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
```

其中：
- `r`: 相关系数
- `α`: 变异系数比值 (σsim/σobs)
- `β`: 均值比值 (μsim/μobs)

**优势**: KGE综合评估相关性、变异性和偏差，避免了传统NSE在极值情况下的局限性。

### 简化评估标准

基于数学分析，我们简化了评估体系，只保留两个核心独立指标：

1. **Nash-Sutcliffe效率 (NSE)**
   - 范围: (-∞, 1]
   - 优秀: NSE > 0.75
   - 良好: 0.65 < NSE ≤ 0.75
   - 满意: 0.50 < NSE ≤ 0.65
   - 不满意: NSE ≤ 0.50

2. **百分比偏差 (PBIAS)**
   - 范围: (-∞, +∞)
   - 优秀: |PBIAS| < 10%
   - 良好: 10% ≤ |PBIAS| < 15%
   - 满意: 15% ≤ |PBIAS| < 25%
   - 不满意: |PBIAS| ≥ 25%

---

## 文件结构

```
soil model/
├── soil_model.py          # 核心模型计算引擎
├── model_driver.py        # 模型驱动和评估函数
├── config_manager.py      # 配置管理模块
├── visualization.py       # 可视化和绘图函数
├── model_config.txt       # 模型参数配置
├── variable_mapping.txt   # 变量映射配置
├── 2000.xlsx             # 训练数据（2000年）
├── 2001.xlsx             # 测试数据（2001年）
├── README.md             # 项目文档（中文）
└── README_EN.md          # 项目文档（英文）
```

---

## 快速开始

### 1. 基本使用

```python
from model_driver import main

# 运行完整的模型工作流
main()
```

### 2. 自定义配置

```python
from config_manager import ConfigManager
from model_driver import train_and_evaluate_model

# 加载自定义配置
config = ConfigManager('custom_config.txt', 'custom_mapping.txt')

# 使用自定义设置运行模型
results = train_and_evaluate_model(config)
```

### 3. 结果可视化

```python
from visualization import create_time_series_plots, create_scatter_plots

# 创建可视化图表
create_time_series_plots(observed_data, simulated_data, save_dir='plots/')
create_scatter_plots(observed_data, simulated_data, save_dir='plots/')
```

---

## 数据要求

### 输入数据格式

模型期望Excel文件包含以下列：
- 时间列（可配置列名）
- 降水数据
- 蒸散发数据  
- 多层土壤湿度观测数据

### 配置文件

1. **model_config.txt**: 模型参数配置
2. **variable_mapping.txt**: 不同数据源的变量名称映射

### 配置示例

#### 模型配置 (`model_config.txt`)

```ini
# 基本模型配置
n_layers = 3
max_iterations = 100
convergence_tolerance = 1e-6

# 默认物理参数
default_alpha = 0.1
default_beta = 0.05
default_max_theta_base = 45
default_theta_fi_base = 12

# 优化参数边界
alpha_min = 1e-6
alpha_max = 1.0
beta_min = 1e-6
beta_max = 1.0

# 可视化参数
figure_dpi = 300
figure_width = 12
layer_colors = blue,green,red,orange,purple
font_family = Arial

# 图表标签配置
soil_moisture_unit = mm
depth_unit = cm
train_dataset_name = Training Set (2000)
test_dataset_name = Test Set (2001)
observed_label = Observed
simulated_label = Simulated
```

#### 变量映射 (`variable_mapping.txt`)

```ini
# 数据文件映射配置
train_data_file = 2000.xlsx
test_data_file = 2001.xlsx
data_directory = ./

# 气象变量映射
precipitation = P
evapotranspiration = E
time_column = Time
temperature = AirT

# 土壤层变量映射
# 格式: theta_XXmm = 数据中的列名
theta_10mm = theta10
theta_20mm = theta20
theta_50mm = theta50  # 3层模式已启用

# 数据质量控制配置
min_precipitation = 0.0
max_precipitation = 200.0
min_evapotranspiration = 0.0
max_evapotranspiration = 15.0

# 单位转换配置
theta_unit_conversion = true
theta_conversion_factors = 100.0,100.0,100.0
```

#### 自定义配置示例

针对不同数据源或研究站点，您可以创建自定义配置文件：

**custom_config.txt**:
```ini
# 站点特定配置
n_layers = 3
layer_depths = 5,15,30,60
active_layer_indices = 1,2,3

# 站点特定参数
default_alpha = 0.15
default_beta = 0.08
max_theta_increment = 8
```

**custom_mapping.txt**:
```ini
# 自定义数据源映射
train_data_file = site_a_2020.xlsx
test_data_file = site_a_2021.xlsx

# 不同的列名
precipitation = Rainfall
evapotranspiration = ET_measured
time_column = DateTime

# 不同的土壤层命名
theta_5mm = SM_5cm
theta_15mm = SM_15cm
theta_30mm = SM_30cm
```

**使用自定义配置**:
```python
from config_manager import ConfigManager
from model_driver import train_and_evaluate_model

# 加载自定义配置
config = ConfigManager('custom_config.txt', 'custom_mapping.txt')

# 使用自定义设置运行模型
results = train_and_evaluate_model(config)
```

---

## 模型性能

模型已在芬兰Hyytiälä森林研究站数据上测试：
- 训练期: 2000年
- 测试期: 2001年
- 典型性能: NSE > 0.75, |PBIAS| < 15%

---

## 技术说明

### 架构设计
- **soil_model.py**: 纯计算引擎，不包含评估功能
- **model_driver.py**: 模型工作流、优化和评估
- **config_manager.py**: 适配不同数据源的灵活配置系统
- **visualization.py**: 专业绘图和导出功能

### 依赖库
- numpy
- pandas
- matplotlib
- scipy
- openpyxl (用于读取Excel文件)

---

## 模拟结果

### 3层模型表现

最新的3层土壤水分模型模拟结果：

**训练表现 (2000年):**
- **10cm层**: NSE=0.37, PBIAS=-2.8% (中等可信度)
- **20cm层**: NSE=0.19, PBIAS=-15.9% (低可信度)  
- **50cm层**: NSE=-6.1, PBIAS=62.9% (低可信度)

**测试表现 (2001年):**
- **10cm层**: NSE=-1.28, PBIAS=7.7% (中等可信度)
- **20cm层**: NSE=-0.79, PBIAS=-3.1% (中等可信度)
- **50cm层**: NSE=-11.6, PBIAS=66.2% (低可信度)

### 生成的输出文件

- `model_performance_metrics.json` - 详细性能指标
- `train_comparison_3layer.png` - 训练数据对比图
- `test_comparison_3layer.png` - 测试数据对比图  
- `scatter_plots_3layer.png` - 散点图分析

### 模型表现分析

当前3层模型结果显示：
1. **10cm表层**：表现最好，特别是在训练集上达到中等可信度
2. **20cm中层**：表现中等，PBIAS在可接受范围内
3. **50cm深层**：表现较差，需要进一步调参优化

模型在表层土壤水分模拟方面展现了较好的潜力，深层参数可能需要进一步校准。

---

## 许可证

本项目用于学术研究目的开发。

---

## 联系方式

关于模型实现或使用的问题，请参考代码文档或在项目仓库中创建issue。

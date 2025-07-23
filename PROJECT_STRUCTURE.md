# 项目结构说明 / Project Structure

## 重组后的文件夹结构 / Reorganized Folder Structure

```
multilayer-soil-water-model/
├── core/                           # 🔧 核心代码文件 / Core Code Files
│   ├── model_driver.py            # 主程序驱动器 / Main Program Driver
│   ├── soil_model.py              # 土壤模型核心算法 / Soil Model Core Algorithm
│   ├── config_manager.py          # 配置管理模块 / Configuration Management Module
│   └── visualization.py           # 可视化模块 / Visualization Module
├── config/                         # ⚙️ 配置文件 / Configuration Files
│   ├── model_config.txt           # 模型参数配置 / Model Parameter Configuration
│   └── variable_mapping.txt       # 变量映射配置 / Variable Mapping Configuration
├── data/                           # 📊 数据文件 / Data Files
│   ├── 2000.xlsx                  # 训练数据（2000年）/ Training Data (Year 2000)
│   └── 2001.xlsx                  # 测试数据（2001年）/ Test Data (Year 2001)
├── docs/                           # 📖 文档文件 / Documentation Files
│   ├── README_CN.md               # 中文详细文档 / Chinese Detailed Documentation
│   └── README_EN.md               # 英文详细文档 / English Detailed Documentation
├── results/                        # 📈 结果文件 / Results Files
│   ├── model_performance_metrics.json    # 性能指标数据 / Performance Metrics Data
│   ├── train_comparison_3layer.png       # 训练对比图 / Training Comparison Chart
│   ├── test_comparison_3layer.png        # 测试对比图 / Test Comparison Chart
│   └── scatter_plots_3layer.png          # 散点图分析 / Scatter Plot Analysis
├── README.md                       # 主文档 / Main Documentation
└── PROJECT_STRUCTURE.md           # 本文件 / This File
```

## 运行方式 / How to Run

### 方法 1：在 core 文件夹中运行 / Method 1: Run from core folder
```bash
cd core
python model_driver.py
```

### 方法 2：从根目录运行 / Method 2: Run from root directory
```bash
python core/model_driver.py
```

## 文件路径更新说明 / File Path Updates

为确保重组后的项目正常运行，已对以下文件进行了路径更新：

### 1. `core/config_manager.py`
- 默认配置文件路径更新为相对路径：
  - `model_config.txt` → `../config/model_config.txt`
  - `variable_mapping.txt` → `../config/variable_mapping.txt`
- `load_data` 函数自动在 `../data/` 文件夹中查找数据文件

### 2. `core/model_driver.py`
- 可视化图片输出路径设置为：`../results/`
- 性能指标JSON文件输出路径设置为：`../results/`
- 更新了配置文件路径提示信息

### 3. `core/visualization.py`
- 所有图片和JSON文件将根据 `output_prefix` 参数保存到指定位置

## 优势 / Benefits

1. **更清晰的项目结构** / **Clearer Project Structure**
   - 代码、配置、数据、文档和结果分离
   - 更容易维护和理解

2. **更好的文件管理** / **Better File Management**
   - 相关文件集中管理
   - 减少根目录文件混乱

3. **更专业的项目组织** / **More Professional Project Organization**
   - 符合标准的软件项目结构
   - 便于版本控制和协作开发

## 注意事项 / Notes

- 所有相对路径都基于 `core` 文件夹作为工作目录
- 如果数据文件不在 `data` 文件夹中，程序会自动回退到原始路径
- 生成的结果文件会自动保存到 `results` 文件夹中

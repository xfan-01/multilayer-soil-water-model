# Multi-layer Soil Water Model Based on Bidirectional Percolation

English Version | [中文版本](./README_CN.md)

## Project Overview

This project implements a physics-based multi-layer soil water dynamics simulation model. The core innovation is **removing artificial inter-layer exchange limitations**, completely based on natural bidirectional percolation flow mechanisms driven by capillary action and gravity in soil physics.

### Key Features
- **Physical Consistency**: Implements natural flow process of "water from wet areas → dry areas"
- **Bidirectional Percolation**: Saturated/unsaturated percolation mechanisms
- **Multi-layer Structure**: Supports arbitrary number of soil profile layers (currently supports 2-3 layers)
- **KGE Optimization**: Uses Kling-Gupta Efficiency as objective function for parameter calibration
- **Simplified Assessment**: Uses NSE and PBIAS as core evaluation metrics, removing redundant indicators
- **Data-driven**: Uses long-term observation data from Hyytiälä Forest Research Station, Finland

---

## Mathematical Foundation

### Core Physical Equations

#### 1. Saturated Percolation
Drainage process when soil water content exceeds saturation value:

```
qs,i = (θi - θs,i) × ks,i    when θi > θs,i
qs,i = 0                     when θi ≤ θs,i
```

**Physical Meaning**: Gravity-driven free drainage, supersaturated water flows directly

#### 2. Unsaturated Percolation
Water movement under combined capillary and gravitational forces:

```
qun,i = (θi - θmin,i) × ku1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^ku2,i
```

**Physical Meaning**: Percolation under unsaturated conditions, intensity has nonlinear relationship with effective water content

#### 3. Capillary Rise (Independent Formula)
To more accurately simulate water movement upward against gravity, we set independent parameters for capillary rise:

```
q_cap,i = (θi - θmin,i) × kc1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^kc2,i
```

#### 4. Bidirectional Percolation Flow
Inter-layer net flow calculation:

```
net_flux = qun_down - q_cap_up
```

Where:
- `qun_down`: Downward percolation from upper layer
- `q_cap_up`: Upward capillary rise from lower layer (using capillary rise formula)

### Variable Symbol Definitions

| Symbol | Meaning | Unit |
|--------|---------|------|
| θi | Water content of layer i | mm |
| θs,i | Saturated water content of layer i | mm |
| θmin,i | Minimum water content of layer i (wilting point) | mm |
| ks,i | Saturated hydraulic conductivity of layer i | mm/day |
| ku1,i | Unsaturated percolation coefficient of layer i | mm/day |
| ku2,i | Unsaturated percolation exponent of layer i | - |
| kc1,i | Capillary rise coefficient of layer i | mm/day |
| kc2,i | Capillary rise exponent of layer i | - |
| P | Precipitation | mm/day |
| E | Evapotranspiration | mm/day |
| R | Surface runoff | mm/day |
| Qgw | Deep percolation | mm/day |

### Water Balance Equation

For layer i soil, water balance is expressed as:

```
Δθi = Pi - Ei - qs,i - qun,i + Qin,i - Qout,i
```

Where:
- `Pi`: Water input received by layer i (precipitation or upper layer drainage)
- `Ei`: Evapotranspiration loss from layer i
- `qs,i`: Saturated drainage from layer i
- `qun,i`: Unsaturated percolation from layer i
- `Qin,i`: Water inflow from adjacent layers
- `Qout,i`: Water outflow to adjacent layers

---

## Optimization and Evaluation System

### KGE Objective Function
The model uses **Kling-Gupta Efficiency (KGE)** as the objective function for parameter optimization:

```
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
```

Where:
- `r`: Correlation coefficient
- `α`: Coefficient of variation ratio (σsim/σobs)
- `β`: Mean ratio (μsim/μobs)

**Advantages**: KGE comprehensively evaluates correlation, variability and bias, avoiding limitations of traditional NSE in extreme cases.

### Simplified Evaluation Standards

Core evaluation metrics:

1. **Nash-Sutcliffe Efficiency (NSE)**
   - Range: (-∞, 1]
   - Excellent: NSE > 0.75
   - Good: 0.65 < NSE ≤ 0.75
   - Satisfactory: 0.50 < NSE ≤ 0.65
   - Unsatisfactory: NSE ≤ 0.50

2. **Percent Bias (PBIAS)**
   - Range: (-∞, +∞)
   - Excellent: |PBIAS| < 10%
   - Good: 10% ≤ |PBIAS| < 15%
   - Satisfactory: 15% ≤ |PBIAS| < 25%
   - Unsatisfactory: |PBIAS| ≥ 25%

---

## File Structure

```
soil model/
├── soil_model.py          # Core model calculation engine
├── model_driver.py        # Model driver and evaluation functions
├── config_manager.py      # Configuration management module
├── visualization.py       # Visualization and plotting functions
├── model_config.txt       # Model parameter configuration
├── variable_mapping.txt   # Variable mapping configuration
├── 2000.xlsx             # Training data (year 2000)
├── 2001.xlsx             # Test data (year 2001)
└── README.md             # Project documentation
```

---

## Quick Start

### 1. Basic Usage

```python
from model_driver import main

# Run complete model workflow
main()
```

### 2. Custom Configuration

```python
from config_manager import ConfigManager
from model_driver import train_and_evaluate_model

# Load custom configuration
config = ConfigManager('custom_config.txt', 'custom_mapping.txt')

# Run model with custom settings
results = train_and_evaluate_model(config)
```

### 3. Visualization

```python
from visualization import create_time_series_plots, create_scatter_plots

# Create visualization plots
create_time_series_plots(observed_data, simulated_data, save_dir='plots/')
create_scatter_plots(observed_data, simulated_data, save_dir='plots/')
```

---

## Data Requirements

### Input Data Format

The model expects Excel files with the following columns:
- Time column (configurable name)
- Precipitation data
- Evapotranspiration data  
- Soil moisture observations for multiple layers

### Configuration Files

1. **model_config.txt**: Model parameters
2. **variable_mapping.txt**: Variable name mappings for different data sources

### Configuration Examples

#### Model Configuration (`model_config.txt`)

```ini
# Basic Model Configuration
n_layers = 3
max_iterations = 100
convergence_tolerance = 1e-6

# Default Physical Parameters
default_alpha = 0.1
default_beta = 0.05
default_max_theta_base = 45
default_theta_fi_base = 12

# Optimization Parameter Boundaries
alpha_min = 1e-6
alpha_max = 1.0
beta_min = 1e-6
beta_max = 1.0

# Visualization Parameters
figure_dpi = 300
figure_width = 12
layer_colors = blue,green,red,orange,purple
font_family = Arial

# Chart Label Configuration
soil_moisture_unit = mm
depth_unit = cm
train_dataset_name = Training Set (2000)
test_dataset_name = Test Set (2001)
observed_label = Observed
simulated_label = Simulated
```

#### Variable Mapping (`variable_mapping.txt`)

```ini
# Data File Mapping Configuration
train_data_file = 2000.xlsx
test_data_file = 2001.xlsx
data_directory = ./

# Meteorological Variable Mapping
precipitation = P
evapotranspiration = E
time_column = Time
temperature = AirT

# Soil Layer Variable Mapping
# Format: theta_XXmm = column_name_in_data
theta_10mm = theta10
theta_20mm = theta20
theta_50mm = theta50  # 3-layer mode enabled

# Data Quality Control Configuration
min_precipitation = 0.0
max_precipitation = 200.0
min_evapotranspiration = 0.0
max_evapotranspiration = 15.0

# Unit Conversion Configuration
theta_unit_conversion = true
theta_conversion_factors = 100.0,100.0,100.0
```

#### Custom Configuration Example

For different data sources or research sites, you can create custom configuration files:

**custom_config.txt**:
```ini
# Site-specific configuration
n_layers = 3
layer_depths = 5,15,30,60
active_layer_indices = 1,2,3

# Site-specific parameters
default_alpha = 0.15
default_beta = 0.08
max_theta_increment = 8
```

**custom_mapping.txt**:
```ini
# Custom data source mapping
train_data_file = site_a_2020.xlsx
test_data_file = site_a_2021.xlsx

# Different column names
precipitation = Rainfall
evapotranspiration = ET_measured
time_column = DateTime

# Different soil layer naming
theta_5mm = SM_5cm
theta_15mm = SM_15cm
theta_30mm = SM_30cm
```

**Usage with custom configuration**:
```python
from config_manager import ConfigManager
from model_driver import train_and_evaluate_model

# Load custom configuration
config = ConfigManager('custom_config.txt', 'custom_mapping.txt')

# Run model with custom settings
results = train_and_evaluate_model(config)
```

---

## Model Performance

The model has been tested on Hyytiälä Forest Research Station data with:
- Training period: 2000
- Test period: 2001
- Typical performance: NSE > 0.75, |PBIAS| < 15%

---

## Technical Notes

### Architecture Design
- **soil_model.py**: Pure calculation engine, no evaluation functions
- **model_driver.py**: Model workflow, optimization, and evaluation
- **config_manager.py**: Flexible configuration system for different data sources
- **visualization.py**: Professional plotting and export functions

### Dependencies
- numpy
- pandas
- matplotlib
- scipy
- openpyxl (for Excel file reading)

---

---

## Simulation Results

### 3-Layer Model Performance

Latest simulation results for 3-layer soil moisture model:

**Training Performance (2000):**
- **10cm layer**: NSE=0.37, PBIAS=-2.8% (Moderate Confidence)
- **20cm layer**: NSE=0.19, PBIAS=-15.9% (Low Confidence)  
- **50cm layer**: NSE=-6.1, PBIAS=62.9% (Low Confidence)

**Test Performance (2001):**
- **10cm layer**: NSE=-1.28, PBIAS=7.7% (Moderate Confidence)
- **20cm layer**: NSE=-0.79, PBIAS=-3.1% (Moderate Confidence)
- **50cm layer**: NSE=-11.6, PBIAS=66.2% (Low Confidence)

### Generated Outputs

- `model_performance_metrics.json` - Detailed performance metrics
- `train_comparison_3layer.png` - Training data comparison plots
- `test_comparison_3layer.png` - Test data comparison plots  
- `scatter_plots_3layer.png` - Scatter plot analysis

### Model Performance Analysis

Current 3-layer model results show:
1. **10cm surface layer**: Best performance, achieving moderate confidence especially on training data
2. **20cm middle layer**: Moderate performance, PBIAS within acceptable range
3. **50cm deep layer**: Poor performance, requires further parameter optimization

The model demonstrates good potential for surface soil moisture simulation, while deeper layer parameters may need further calibration.

---

## License

This project is developed for academic research purposes.

---

## Contact

For questions about the model implementation or usage, please refer to the code documentation or create an issue in the project repository.

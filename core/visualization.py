#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization Module - Configuration-driven Dynamic Chart Generation
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def setup_plotting(config_manager=None):
    """Setup matplotlib parameters"""
    if config_manager:
        font_family = config_manager.get('font_family', 'SimHei')
        figure_dpi = config_manager.get('figure_dpi', 300)
    else:
        font_family = 'SimHei'
        figure_dpi = 300
        
    plt.rcParams['font.sans-serif'] = [font_family]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = figure_dpi


def get_layer_colors(n_layers, config_manager=None):
    """Get color scheme based on number of layers - use predefined color set"""
    # Predefined high-quality color set
    default_colors = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf'
    ]
    
    # Ensure sufficient colors
    while len(default_colors) < n_layers:
        default_colors.extend(default_colors)
    
    return default_colors[:n_layers]


def get_labels(config_manager):
    """Get all labels from configuration manager"""
    if not config_manager:
        return {
            'soil_moisture_unit': 'mm',
            'depth_unit': 'cm', 
            'train_dataset_name': 'Training Set (2000)',
            'test_dataset_name': 'Test Set (2001)',
            'observed_label': 'Observed',
            'simulated_label': 'Simulated',
            'date_label': 'Date',
            'soil_moisture_label': 'Soil Moisture'
        }
    
    return {
        'soil_moisture_unit': config_manager.get('soil_moisture_unit', 'mm'),
        'depth_unit': config_manager.get('depth_unit', 'cm'),
        'train_dataset_name': config_manager.get('train_dataset_name', 'Training Set'),
        'test_dataset_name': config_manager.get('test_dataset_name', 'Test Set'),
        'observed_label': config_manager.get('observed_label', 'Observed'),
        'simulated_label': config_manager.get('simulated_label', 'Simulated'),
        'date_label': config_manager.get('date_label', 'Date'),
        'soil_moisture_label': config_manager.get('soil_moisture_label', 'Soil Moisture')
    }


def create_time_series_plots(train_data, test_data, results, config_manager=None, output_prefix=''):
    """
    Create time series comparison plots - configuration-driven dynamic adaptation
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    results : dict
        Model results
    config_manager : ConfigManager
        Configuration manager
    output_prefix : str
        Output file prefix
    """
    # Get label configuration
    labels = get_labels(config_manager)
    
    # Get layer information - use theta_layers instead of layer_info
    if results.get('theta_layers'):
        depth_labels = [f"{layer['depth']}{labels['depth_unit']}" for layer in results['theta_layers']]
        n_layers = len(results['theta_layers'])
        layer_columns = [layer['output_column'] for layer in results['theta_layers']]
    else:
        # Fallback solution
        n_layers = len(results['layer_columns'])
        layer_columns = results['layer_columns']
        depth_labels = []
        for col in layer_columns:
            if 'theta10' in col or '_10' in col:
                depth_labels.append(f"10{labels['depth_unit']}")
            elif 'theta20' in col or '_20' in col:
                depth_labels.append(f"20{labels['depth_unit']}")
            elif 'theta50' in col or '_50' in col:
                depth_labels.append(f"50{labels['depth_unit']}")
            else:
                depth_labels.append(col.replace('_mm', '').replace('theta', '') + labels['depth_unit'])
    
    colors = get_layer_colors(n_layers, config_manager)
    
    # 1. Training set time series comparison
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 3*n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]
    
    for i, depth in enumerate(depth_labels):
        ax = axes[i]
        
        # Observed values
        ax.plot(train_data.index, train_data[layer_columns[i]], 
                color=colors[i], label=f'{labels["observed_label"]}({depth})', linewidth=2, alpha=0.8)
        
        # Simulated values
        ax.plot(train_data.index, results['train_simulated'][:, i], 
                color=colors[i], linestyle='--', label=f'{labels["simulated_label"]}({depth})', linewidth=2)
        
        # Performance metrics - show NSE and PBIAS
        if results.get('theta_layers'):
            layer_key = f"layer{results['theta_layers'][i]['depth']}"
        else:
            layer_key = f'layer{depth.replace(labels["depth_unit"], "")}'
        
        nse = results['train_performance'][layer_key]['NSE']
        pbias = results['train_performance'][layer_key]['PBIAS']
        
        ax.text(0.02, 0.95, 
                f'NSE={nse:.3f}, PBIAS={pbias:.1f}%', 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10)
        
        ax.set_ylabel(f'{depth} {labels["soil_moisture_label"]} ({labels["soil_moisture_unit"]})', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set Y-axis range
        all_values = np.concatenate([
            train_data[layer_columns[i]].dropna().values,
            results['train_simulated'][:, i]
        ])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    axes[-1].set_xlabel(labels['date_label'], fontsize=11)
    fig.suptitle(f'{labels["train_dataset_name"]} - {n_layers}-Layer {labels["soil_moisture_label"]} Simulation Results', fontsize=14, fontweight='bold')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}train_comparison_{n_layers}layer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Test set time series comparison
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 3*n_layers), sharex=True)
    if n_layers == 1:
        axes = [axes]
    
    for i, depth in enumerate(depth_labels):
        ax = axes[i]
        
        # Observed values
        ax.plot(test_data.index, test_data[layer_columns[i]], 
                color=colors[i], label=f'{labels["observed_label"]}({depth})', linewidth=2, alpha=0.8)
        
        # Simulated values
        ax.plot(test_data.index, results['test_simulated'][:, i], 
                color=colors[i], linestyle='--', label=f'{labels["simulated_label"]}({depth})', linewidth=2)
        
        # Performance metrics
        if results.get('theta_layers'):
            layer_key = f"layer{results['theta_layers'][i]['depth']}"
        else:
            layer_key = f'layer{depth.replace(labels["depth_unit"], "")}'
            
        nse = results['test_performance'][layer_key]['NSE']
        pbias = results['test_performance'][layer_key]['PBIAS']
        
        ax.text(0.02, 0.95, 
                f'NSE={nse:.3f}, PBIAS={pbias:.1f}%', 
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10)
        
        ax.set_ylabel(f'{depth} {labels["soil_moisture_label"]} ({labels["soil_moisture_unit"]})', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set Y-axis range
        all_values = np.concatenate([
            test_data[layer_columns[i]].dropna().values,
            results['test_simulated'][:, i]
        ])
        y_min, y_max = np.min(all_values), np.max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
    
    axes[-1].set_xlabel(labels['date_label'], fontsize=11)
    fig.suptitle(f'{labels["test_dataset_name"]} - {n_layers}-Layer {labels["soil_moisture_label"]} Simulation Results', fontsize=14, fontweight='bold')
    
    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}test_comparison_{n_layers}layer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {n_layers}-layer time series comparison plot")


def create_scatter_plots(train_data, test_data, results, config_manager=None, output_prefix=''):
    """
    Create scatter comparison plots - configuration-driven dynamic adaptation
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    results : dict
        Model results
    config_manager : ConfigManager
        Configuration manager
    output_prefix : str
        Output file prefix
    """
    # Get label configuration
    labels = get_labels(config_manager)
    
    # Get layer information
    if results.get('theta_layers'):
        depth_labels = [f"{layer['depth']}{labels['depth_unit']}" for layer in results['theta_layers']]
        n_layers = len(results['theta_layers'])
        layer_columns = [layer['output_column'] for layer in results['theta_layers']]
    else:
        n_layers = len(results['layer_columns'])
        layer_columns = results['layer_columns']
        depth_labels = []
        for col in layer_columns:
            if 'theta10' in col or '_10' in col:
                depth_labels.append(f"10{labels['depth_unit']}")
            elif 'theta20' in col or '_20' in col:
                depth_labels.append(f"20{labels['depth_unit']}")
            elif 'theta50' in col or '_50' in col:
                depth_labels.append(f"50{labels['depth_unit']}")
            else:
                depth_labels.append(col.replace('_mm', '').replace('theta', '') + labels['depth_unit'])
    
    colors = get_layer_colors(n_layers, config_manager)
    
    # Dynamic subplot layout calculation - training and test sets shown separately
    ncols = n_layers
    nrows = 2  # First row for training set, second row for test set
    
    # Calculate figure size - adjust based on actual number of layers
    fig_width = 5 * ncols
    fig_height = 10  # Fixed height, two rows
    
    # Create scatter plots (training and test sets shown separately)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    # Handle single layer case
    if n_layers == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle(f'{n_layers}-Layer Simulated vs {labels["observed_label"]} Scatter Plot Comparison', 
                fontsize=14, fontweight='bold')
    
    for i, depth in enumerate(depth_labels):
        # Training set scatter plot (first row)
        ax_train = axes[0, i]
        
        # Test set scatter plot (second row)  
        ax_test = axes[1, i]
        
        # Training set data
        train_obs = train_data[layer_columns[i]].values
        train_sim = results['train_simulated'][:, i]
        
        # Test set data
        test_obs = test_data[layer_columns[i]].values
        test_sim = results['test_simulated'][:, i]
        
        # Performance metrics
        if results.get('theta_layers'):
            layer_key = f"layer{results['theta_layers'][i]['depth']}"
        else:
            layer_key = f'layer{depth.replace(labels["depth_unit"], "")}'
        
        train_nse = results['train_performance'][layer_key]['NSE']
        train_pbias = results['train_performance'][layer_key]['PBIAS']
        test_nse = results['test_performance'][layer_key]['NSE']
        test_pbias = results['test_performance'][layer_key]['PBIAS']
        
        # === Training set scatter plot ===
        ax_train.scatter(train_obs, train_sim, alpha=0.6, color=colors[i], s=30, 
                        edgecolors='white', linewidth=0.5)
        
        # Calculate training set 1:1 line range
        train_min = min(train_obs.min(), train_sim.min())
        train_max = max(train_obs.max(), train_sim.max())
        ax_train.plot([train_min, train_max], [train_min, train_max], 'k--', lw=2, alpha=0.8, label='1:1 Line')
        
        # Training set statistics
        train_stats_text = f'NSE={train_nse:.3f}, PBIAS={train_pbias:.1f}%'
        ax_train.text(0.05, 0.95, train_stats_text, 
                     transform=ax_train.transAxes, verticalalignment='top',
                     bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        ax_train.set_xlabel(f'{labels["observed_label"]} ({depth}) ({labels["soil_moisture_unit"]})', fontsize=11)
        ax_train.set_ylabel(f'{labels["simulated_label"]} ({depth}) ({labels["soil_moisture_unit"]})', fontsize=11)
        ax_train.set_title(f'{labels["train_dataset_name"]} - {depth}', fontsize=12, fontweight='bold')
        ax_train.grid(True, alpha=0.3)
        ax_train.legend(fontsize=9)
        
        # Set equal axis ranges
        ax_train.set_xlim(train_min, train_max)
        ax_train.set_ylim(train_min, train_max)
        ax_train.set_aspect('equal', adjustable='box')
        
        # === Test set scatter plot ===
        ax_test.scatter(test_obs, test_sim, alpha=0.6, color=colors[i], s=30, 
                       edgecolors='black', linewidth=0.5, marker='^')
        
        # Calculate test set 1:1 line range
        test_min = min(test_obs.min(), test_sim.min())
        test_max = max(test_obs.max(), test_sim.max())
        ax_test.plot([test_min, test_max], [test_min, test_max], 'k--', lw=2, alpha=0.8, label='1:1 Line')
        
        # Test set statistics
        test_stats_text = f'NSE={test_nse:.3f}, PBIAS={test_pbias:.1f}%'
        ax_test.text(0.05, 0.95, test_stats_text, 
                    transform=ax_test.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        ax_test.set_xlabel(f'{labels["observed_label"]} ({depth}) ({labels["soil_moisture_unit"]})', fontsize=11)
        ax_test.set_ylabel(f'{labels["simulated_label"]} ({depth}) ({labels["soil_moisture_unit"]})', fontsize=11)
        ax_test.set_title(f'{labels["test_dataset_name"]} - {depth}', fontsize=12, fontweight='bold')
        ax_test.grid(True, alpha=0.3)
        ax_test.legend(fontsize=9)
        
        # Set equal axis ranges
        ax_test.set_xlim(test_min, test_max)
        ax_test.set_ylim(test_min, test_max)
        ax_test.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}scatter_plots_{n_layers}layer.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Generated {n_layers}-layer scatter comparison plot")


def generate_all_plots(train_data, test_data, results, config_manager=None, output_prefix=''):
    """
    Generate all visualization charts - configuration-driven
    
    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data
    test_data : pd.DataFrame
        Test data
    results : dict
        Model results
    config_manager : ConfigManager
        Configuration manager
    output_prefix : str
        Output file prefix
    """
    setup_plotting(config_manager)
    
    # Get layer information
    if results.get('theta_layers'):
        depth_labels = [f"{layer['depth']}cm" for layer in results['theta_layers']]
        n_layers = len(results['theta_layers'])
    else:
        n_layers = len(results['layer_columns'])
        depth_labels = []
        for col in results['layer_columns']:
            # Simplified depth extraction logic
            if '_10' in col or 'theta10' in col:
                depth_labels.append('10cm')
            elif '_20' in col or 'theta20' in col:
                depth_labels.append('20cm')
            elif '_50' in col or 'theta50' in col:
                depth_labels.append('50cm')
            else:
                # General number extraction
                import re
                match = re.search(r'(\d+)', col)
                if match:
                    depth_labels.append(f"{match.group(1)}cm")
                else:
                    depth_labels.append(col)
    
    print(f"\n=== Starting {n_layers}-layer visualization chart generation ===")
    print(f"Layer depths: {depth_labels}")
    
    # Generate time series plots
    create_time_series_plots(train_data, test_data, results, config_manager, output_prefix)
    
    # Generate scatter plots
    create_scatter_plots(train_data, test_data, results, config_manager, output_prefix)
    
    print(f"\n=== {n_layers}-layer visualization chart generation completed ===")
    print(f"Generated files:")
    print(f"  - {output_prefix}train_comparison_{n_layers}layer.png")
    print(f"  - {output_prefix}test_comparison_{n_layers}layer.png") 
    print(f"  - {output_prefix}scatter_plots_{n_layers}layer.png")
    
    return {
        'n_layers': n_layers,
        'depth_labels': depth_labels,
        'files_generated': [
            f"{output_prefix}train_comparison_{n_layers}layer.png",
            f"{output_prefix}test_comparison_{n_layers}layer.png",
            f"{output_prefix}scatter_plots_{n_layers}layer.png"
        ]
    }


def export_performance_metrics(train_detailed_perf, test_detailed_perf, layer_info=None, output_prefix=''):
    """
    Export performance metrics to file and generate visualization charts
    
    Parameters:
    -----------
    train_detailed_perf : dict
        Training set detailed performance metrics
    test_detailed_perf : dict  
        Test set detailed performance metrics
    layer_info : list, optional
        Layer information list
    output_prefix : str
        Output file prefix
        
    Returns:
    --------
    dict : Generated file information
    """
    import pandas as pd
    import json
    from datetime import datetime
    
    # Prepare data
    metrics_data = []
    layer_names = list(train_detailed_perf.keys())
    
    for layer_name in layer_names:
        train_metrics = train_detailed_perf[layer_name]
        test_metrics = test_detailed_perf[layer_name]
        
        # Extract depth information
        if layer_info:
            depth_info = next((layer for layer in layer_info if f"{layer['depth']}cm" == layer_name), None)
            depth = depth_info['depth'] if depth_info else layer_name.replace('cm', '')
        else:
            depth = layer_name.replace('cm', '')
        
        metrics_data.append({
            'Layer': layer_name,
            'Depth_cm': depth,
            'Train_NSE': train_metrics.get('NSE', np.nan),
            'Train_PBIAS': train_metrics.get('PBIAS', np.nan),
            'Train_R2': train_metrics.get('R²', np.nan),  # For reference only
            'Train_Status': train_metrics.get('status', 'Unknown'),
            'Test_NSE': test_metrics.get('NSE', np.nan),
            'Test_PBIAS': test_metrics.get('PBIAS', np.nan),
            'Test_R2': test_metrics.get('R²', np.nan),  # For reference only
            'Test_Status': test_metrics.get('status', 'Unknown'),
        })
    
    # Create DataFrame
    df = pd.DataFrame(metrics_data)
    
    # 1. Export to Excel file
    excel_filename = f"{output_prefix}model_performance_metrics.xlsx"
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main metrics table - keep only NSE and PBIAS
        main_metrics = df[['Layer', 'Depth_cm', 'Train_NSE', 'Train_PBIAS', 
                          'Test_NSE', 'Test_PBIAS', 'Train_Status', 'Test_Status']]
        main_metrics.to_excel(writer, sheet_name='Main Metrics', index=False)
        
        # Detailed metrics table (including R² for reference)
        df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        
        # Evaluation criteria explanation - simplified to core metrics
        criteria_df = pd.DataFrame({
            'Metric': ['NSE', 'PBIAS'],
            'Acceptable Standard': ['> 0.5', '≤ ±15%'],
            'Description': [
                'Nash-Sutcliffe Efficiency: Measures model prediction accuracy',
                'Percent Bias: Measures systematic bias'
            ]
        })
        criteria_df.to_excel(writer, sheet_name='Evaluation Criteria', index=False)
    
    # 2. Export to JSON file
    json_filename = f"{output_prefix}model_performance_metrics.json"
    
    # Convert NumPy data types to JSON serializable types
    def convert_to_json_serializable(obj):
        """Recursively convert object to JSON serializable type"""
        if isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    export_data = {
        'export_time': datetime.now().isoformat(),
        'evaluation_criteria': {
            'NSE': {'threshold': 0.5, 'condition': '>', 'description': 'Nash-Sutcliffe Efficiency'},
            'PBIAS': {'threshold': 15, 'condition': '≤±', 'description': 'Percent Bias'}
        },
        'train_performance': convert_to_json_serializable(train_detailed_perf),
        'test_performance': convert_to_json_serializable(test_detailed_perf)
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation metrics exported:")
    print(f"   Excel file: {excel_filename}")
    print(f"   JSON file: {json_filename}")
    
    return {
        'excel_file': excel_filename,
        'json_file': json_filename,
        'summary': {
            'total_layers': len(layer_names),
            'train_reliable_layers': sum(1 for m in train_detailed_perf.values() if m.get('status') == 'Satisfactory'),
            'test_reliable_layers': sum(1 for m in test_detailed_perf.values() if m.get('status') == 'Satisfactory')
        }
    }

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Soil Moisture Model Driver - Configuration-driven Main Control Program
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Import custom modules
from config_manager import ConfigManager, load_data
from soil_model import SoilModel, simulate, calibrate
from visualization import generate_all_plots


def evaluate(observed, simulated):
    """
    Simplified model performance evaluation function
    
    Parameters:
    ----------
    observed : array_like
        Observed values
    simulated : array_like
        Simulated values
        
    Returns:
    -------
    dict : Simplified performance metrics dictionary (only NSE and PBIAS)
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(observed) | np.isnan(simulated))
    if np.sum(valid_mask) == 0:
        return {'NSE': np.nan, 'PBIAS': np.nan}
    
    obs_valid = observed[valid_mask]
    sim_valid = simulated[valid_mask]
    
    # Calculate metrics
    nse = 1 - np.sum((obs_valid - sim_valid)**2) / np.sum((obs_valid - np.mean(obs_valid))**2)
    pbias = 100 * np.sum(obs_valid - sim_valid) / np.sum(obs_valid)
    
    return {
        'NSE': nse,
        'PBIAS': pbias
    }


def evaluate_model_performance(observed, simulated, layer_names=None):
    """
    Calculate standard evaluation metrics for hydrological model
    
    Based on Moriasi et al. (2007) and other hydrological literature recommendations for soil moisture models:
    - NSE > 0.5: Satisfactory  
    - PBIAS ≤ ±15%: Satisfactory
    
    Parameters:
    ----------
    observed : array_like, shape (n_timesteps, n_layers)
        Observed soil moisture data
    simulated : array_like, shape (n_timesteps, n_layers)  
        Simulated soil moisture data
    layer_names : list, optional
        Layer name list, e.g. ['5cm', '15cm', '30cm']
        
    Returns:
    -------
    dict : Detailed evaluation metrics for each layer
    """
    if layer_names is None:
        layer_names = [f"Layer_{i+1}" for i in range(observed.shape[1])]
    
    results = {}
    
    print("=" * 80)
    print("Soil Moisture Model Performance Evaluation")
    print("=" * 80)
    print("Evaluation Criteria (Two Core Metrics):")
    print("  NSE > 0.5       (Nash-Sutcliffe Efficiency - Prediction Accuracy)")
    print("  PBIAS ≤ ±15%   (Percent Bias - Systematic Bias)")
    print("-" * 80)
    
    for i, layer_name in enumerate(layer_names):
        obs = observed[:, i]
        sim = simulated[:, i]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(obs) | np.isnan(sim))
        if not np.any(valid_mask) or len(obs[valid_mask]) < 3:
            print(f"Warning: {layer_name:8s}: Insufficient data for metric calculation")
            results[layer_name] = {
                'NSE': np.nan, 'PBIAS': np.nan,
                'status': 'Insufficient data'
            }
            continue
            
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]
        
        # Nash-Sutcliffe efficiency (NSE)
        obs_mean = np.mean(obs_valid)
        nse = 1 - np.sum((obs_valid - sim_valid)**2) / np.sum((obs_valid - obs_mean)**2)
        
        # Percent bias (PBIAS)
        pbias = 100 * np.sum(obs_valid - sim_valid) / np.sum(obs_valid)
        
        # Reliability assessment (based on NSE and PBIAS only)
        nse_status = "Pass" if nse > 0.5 else "Fail"
        pbias_status = "Pass" if abs(pbias) <= 15 else "Fail"
        
        # Overall assessment (based on 2 core metrics)
        passed_criteria = sum([nse > 0.5, abs(pbias) <= 15])
        if passed_criteria >= 2:
            overall_status = "Satisfactory"
        elif passed_criteria >= 1:
            overall_status = "Fair"
        else:
            overall_status = "Unsatisfactory"
        
        # Output results
        print(f"{layer_name:8s}:")
        print(f"  NSE  = {nse:6.3f} {nse_status:>4s}     PBIAS= {pbias:6.1f}% {pbias_status:>4s}")
        print(f"  Overall: {overall_status} ({passed_criteria}/2 core metrics passed)")
        print()
        
        # Save results
        results[layer_name] = {
            'NSE': nse, 
            'PBIAS': pbias,
            'passed_criteria': passed_criteria,
            'status': overall_status
        }
    
    # Overall statistics
    all_nse = [results[layer]['NSE'] for layer in results if not np.isnan(results[layer]['NSE'])]
    all_pbias = [results[layer]['PBIAS'] for layer in results if not np.isnan(results[layer]['PBIAS'])]
    
    if all_nse:
        print("-" * 80)
        print("Overall Statistics:")
        print(f"  Average NSE  = {np.mean(all_nse):.3f} ± {np.std(all_nse):.3f}")
        print(f"  Average PBIAS= {np.mean(all_pbias):.1f}% ± {np.std(all_pbias):.1f}%")
        
        # Reliable layer statistics
        reliable_layers = sum(1 for layer in results if 
                            results[layer]['status'] == "Satisfactory")
        total_layers = len([layer for layer in results if 
                          not np.isnan(results[layer]['NSE'])])
        print(f"  Reliable layers: {reliable_layers}/{total_layers}")
        print("=" * 80)
    
    return results

def evaluate_model_performance_detailed(observed, simulated, layer_names=None, year_label="Unknown"):
    """
    Calculate detailed evaluation metrics for each layer separately
    
    Parameters:
    ----------
    observed : array_like, shape (n_timesteps, n_layers)
        Observed soil moisture data
    simulated : array_like, shape (n_timesteps, n_layers)  
        Simulated soil moisture data
    layer_names : list, optional
        Layer name list, e.g. ['10cm', '20cm', '50cm']
    year_label : str, optional
        Year label for output display
        
    Returns:
    -------
    dict : Detailed evaluation metrics for each layer
    """
    if layer_names is None:
        layer_names = [f"Layer_{i+1}" for i in range(observed.shape[1])]
    
    results = {}
    
    print("=" * 90)
    print(f"Detailed Model Performance Evaluation - {year_label}")
    print("=" * 90)
    print("Individual Layer Evaluation Metrics:")
    print("  NSE > 0.5       = Satisfactory Prediction Accuracy")
    print("  PBIAS ≤ ±15%   = Satisfactory Systematic Bias")
    print("=" * 90)
    
    for i, layer_name in enumerate(layer_names):
        obs = observed[:, i]
        sim = simulated[:, i]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(obs) | np.isnan(sim))
        if not np.any(valid_mask) or len(obs[valid_mask]) < 3:
            print(f"Warning: {layer_name}: Insufficient data for metric calculation")
            results[layer_name] = {
                'NSE': np.nan, 'PBIAS': np.nan,
                'nse_status': 'No Data', 'pbias_status': 'No Data',
                'overall_status': 'Insufficient data'
            }
            continue
            
        obs_valid = obs[valid_mask]
        sim_valid = sim[valid_mask]
        
        # Calculate metrics (only NSE and PBIAS)
        obs_mean = np.mean(obs_valid)
        nse = 1 - np.sum((obs_valid - sim_valid)**2) / np.sum((obs_valid - obs_mean)**2)
        pbias = 100 * np.sum(obs_valid - sim_valid) / np.sum(obs_valid)
        
        # Individual metric assessment
        nse_status = "Satisfactory" if nse > 0.5 else "Unsatisfactory"
        pbias_status = "Satisfactory" if abs(pbias) <= 15 else "Unsatisfactory"
        
        # Calculate confidence level based on both core metrics
        passed_criteria = sum([nse > 0.5, abs(pbias) <= 15])
        if passed_criteria >= 2:
            overall_status = "High Confidence"
            confidence_percent = "≥90%"
        elif passed_criteria >= 1:
            overall_status = "Moderate Confidence"  
            confidence_percent = "60-90%"
        else:
            overall_status = "Low Confidence"
            confidence_percent = "<60%"
        
        # Detailed output for each layer
        print(f"\n📊 {layer_name} Layer Analysis:")
        print(f"   ├─ NSE (Nash-Sutcliffe Efficiency)")
        print(f"   │  ├─ Value: {nse:6.3f}")
        print(f"   │  ├─ Status: {nse_status}")
        print(f"   │  └─ Interpretation: {'Good prediction accuracy' if nse > 0.5 else 'Poor prediction accuracy'}")
        print(f"   │")
        print(f"   ├─ PBIAS (Percent Bias)")
        print(f"   │  ├─ Value: {pbias:6.1f}%")
        print(f"   │  ├─ Status: {pbias_status}")
        print(f"   │  └─ Interpretation: {'Low systematic bias' if abs(pbias) <= 15 else 'High systematic bias'}")
        print(f"   │")
        print(f"   └─ 🎯 Overall Assessment:")
        print(f"      ├─ Confidence Level: {overall_status}")
        print(f"      ├─ Reliability: {confidence_percent}")
        print(f"      └─ Core Metrics Passed: {passed_criteria}/2")
        
        # Save results (removed R²)
        results[layer_name] = {
            'NSE': nse, 
            'PBIAS': pbias,
            'nse_status': nse_status,
            'pbias_status': pbias_status,
            'overall_status': overall_status,
            'confidence_percent': confidence_percent,
            'passed_criteria': passed_criteria
        }
    
    print("\n" + "=" * 90)
    print(f"Summary for {year_label}:")
    
    # Calculate summary statistics
    valid_layers = [layer for layer in results if not np.isnan(results[layer]['NSE'])]
    if valid_layers:
        high_confidence = sum(1 for layer in valid_layers if results[layer]['overall_status'] == "High Confidence")
        moderate_confidence = sum(1 for layer in valid_layers if results[layer]['overall_status'] == "Moderate Confidence")
        low_confidence = sum(1 for layer in valid_layers if results[layer]['overall_status'] == "Low Confidence")
        
        print(f"  Total Layers: {len(valid_layers)}")
        print(f"  High Confidence: {high_confidence} layers")
        print(f"  Moderate Confidence: {moderate_confidence} layers")
        print(f"  Low Confidence: {low_confidence} layers")
        
        # Average metrics (removed R²)
        avg_nse = np.mean([results[layer]['NSE'] for layer in valid_layers])
        avg_pbias = np.mean([results[layer]['PBIAS'] for layer in valid_layers])
        
        print(f"  Average NSE: {avg_nse:.3f}")
        print(f"  Average PBIAS: {avg_pbias:.1f}%")
    
    print("=" * 90)
    
    return results


def prepare_data(observed_data, layer_columns):
    """
    Prepare training data and handle NaN values
    
    Parameters:
    ----------
    observed_data : pd.DataFrame
        Observed data
    layer_columns : list
        Layer column name list
        
    Returns:
    -------
    tuple : (precipitation, evaporation, observed soil moisture)
    """
    # Extract input data
    precip = observed_data['Precip'].values
    evap = observed_data['Evap'].values
    
    # Create observed soil moisture matrix
    observed_moisture = np.column_stack([observed_data[col].values for col in layer_columns])
    
    # Handle NaN values
    precip = np.nan_to_num(precip, nan=0.0)
    evap = np.nan_to_num(evap, nan=0.0)
    
    # Handle NaN values for each layer of observed data through interpolation
    for i in range(observed_moisture.shape[1]):
        col_data = observed_moisture[:, i]
        mask = np.isnan(col_data)
        if np.any(mask):
            col_data[mask] = np.interp(
                np.flatnonzero(mask), 
                np.flatnonzero(~mask), 
                col_data[~mask]
            )
        observed_moisture[:, i] = col_data
    
    return precip, evap, observed_moisture


def train_and_evaluate_model(model, train_data, test_data, layer_info=None):
    """
    Complete model training and evaluation workflow
    
    Parameters:
    ----------
    model : SoilModel
        Soil model instance
    train_data : pd.DataFrame 
        Training data
    test_data : pd.DataFrame
        Testing data
    layer_info : list, optional
        Layer information configuration
        
    Returns:
    -------
    dict : Complete training and evaluation results
    """
    print("\n=== Starting model training and evaluation ===")
    
    # Determine layer column names
    if layer_info is not None:
        layer_columns = [layer['output_column'] for layer in layer_info]
        print(f"Using configuration-driven layer mapping:")
        for i, layer in enumerate(layer_info):
            print(f"  Model layer {i+1} -> Depth {layer['depth']}cm -> Column '{layer['output_column']}'")
    else:
        # Fallback to original logic
        layer_columns = []
        if model.n_layers >= 1 and 'theta_10mm' in train_data.columns:
            layer_columns.append('theta_10mm')
        if model.n_layers >= 2 and 'theta_20mm' in train_data.columns:
            layer_columns.append('theta_20mm')
        if model.n_layers >= 3 and 'theta_50mm' in train_data.columns:
            layer_columns.append('theta_50mm')
    
    # Ensure layer count matches
    if len(layer_columns) != model.n_layers:
        print(f"Warning: Model layers ({model.n_layers}) do not match available data columns ({len(layer_columns)})")
        print(f"Available columns: {layer_columns}")
        
        # Adjust model layers to match data
        model.n_layers = len(layer_columns)
        model.layers = [{"theta": 0.0} for _ in range(model.n_layers)]
        # Regenerate parameters to match new layer count
        model.default_params = model.get_default_params()
        print(f"Adjusted model layer count to: {model.n_layers}")
        print(f"Regenerated {model.n_layers} layer parameters")
    
    # Prepare training data
    train_precip, train_evap, train_observed = prepare_data(train_data, layer_columns)
    
    print(f"Training data check:")
    print(f"  NaN values in precipitation data: {np.sum(np.isnan(train_precip))}")
    print(f"  NaN values in evaporation data: {np.sum(np.isnan(train_evap))}")
    print(f"  NaN values in observed data: {np.sum(np.isnan(train_observed))}")
    print(f"  Model layers: {model.n_layers}, Data layers: {train_observed.shape[1]}")
    
    # 1. Parameter calibration
    print("\n1. Starting parameter calibration...")
    optimized_params = calibrate(model, train_precip, train_evap, train_observed)
    
    # Apply optimized parameters (using new parameter structure)
    model.set_params({
        'k_s1': optimized_params['k_s1'],    # Saturated conductivity coefficient
        'k_u1': optimized_params['k_u1'],    # Unsaturated conductivity coefficient
        'k_c1': optimized_params['k_c1']     # Capillary rise coefficient
    })
    
    print(f"   Parameter calibration completed! Final RMSE: {optimized_params['optimization_result'].fun:.4f}")
    print(f"   Saturated conductivity (k_s1): {[f'{k:.4f}' for k in optimized_params['k_s1']]}")
    print(f"   Unsaturated conductivity (k_u1): {[f'{k:.4f}' for k in optimized_params['k_u1']]}")
    print(f"   Capillary rise coefficient (k_c1): {[f'{k:.4f}' for k in optimized_params['k_c1']]}")
    
    # 2. Training set simulation
    print("2. Running training set simulation...")
    model.reset(initial_theta=train_observed[0])
    train_simulated, R_train, Qdrain_train, Qexchange_train, Qgw_train = simulate(
        model, train_precip, train_evap
    )
    
    # 3. Prepare test data
    test_precip, test_evap, test_observed = prepare_data(test_data, layer_columns)
    
    print("3. Running test set simulation...")
    model.reset(initial_theta=test_observed[0])
    test_simulated, R_test, Qdrain_test, Qexchange_test, Qgw_test = simulate(
        model, test_precip, test_evap
    )
    
    # 4. Performance evaluation
    print("4. Conducting performance evaluation...")
    
    # Dynamically generate layer names
    if layer_info is not None:
        layer_names = [f"layer{layer['depth']}" for layer in layer_info]  # Keep original format: layer10, layer20, layer50
    else:
        layer_names = ['layer10', 'layer20', 'layer50'][:model.n_layers]
    
    # Create depth format layer names for detailed evaluation
    detailed_layer_names = []
    if layer_info is not None:
        detailed_layer_names = [f"{layer['depth']}cm" for layer in layer_info]
    else:
        detailed_layer_names = ['10cm', '20cm', '50cm'][:model.n_layers]
    
    # Use new detailed evaluation metric function
    print("\n=== Training Set Detailed Evaluation ===")
    train_detailed_perf = evaluate_model_performance_detailed(
        train_observed, train_simulated, detailed_layer_names, "Training 2000"
    )
    
    print("\n=== Test Set Detailed Evaluation ===")
    test_detailed_perf = evaluate_model_performance_detailed(
        test_observed, test_simulated, detailed_layer_names, "Testing 2001"
    )
    
    # Maintain backward compatible simple metrics
    train_perf = {}
    test_perf = {}
    
    for i, layer_name in enumerate(layer_names):
        train_perf[layer_name] = evaluate(
            train_observed[:, i], train_simulated[:, i]
        )
        test_perf[layer_name] = evaluate(
            test_observed[:, i], test_simulated[:, i]
        )
    
    print("=== Training and Evaluation Completed ===")
    
    return {
        'optimized_params': optimized_params,
        'train_simulated': train_simulated,
        'test_simulated': test_simulated,
        'train_performance': train_perf,
        'test_performance': test_perf,
        'train_detailed_performance': train_detailed_perf,  # Added detailed evaluation
        'test_detailed_performance': test_detailed_perf,    # Added detailed evaluation
        'layer_columns': layer_columns,
        'theta_layers': layer_info,
        'intermediate_vars': {
            'R_train': R_train, 'Qdrain_train': Qdrain_train,
            'Qexchange_train': Qexchange_train, 'Qgw_train': Qgw_train,
            'R_test': R_test, 'Qdrain_test': Qdrain_test,
            'Qexchange_test': Qexchange_test, 'Qgw_test': Qgw_test
        }
    }


def print_results(results):
    """
    Print model results summary
    
    Parameters:
    ----------
    results : dict
        Model results
    """
    print("\n" + "="*60)
    print("Model Results Summary")
    print("="*60)
    
    # Training set performance
    print("\n=== Training Set Performance ===")
    for layer, perf in results['train_performance'].items():
        print(f"{layer}: NSE={perf['NSE']:.3f}, PBIAS={perf['PBIAS']:.1f}%")
    
    # Test set performance
    print("\n=== Test Set Performance ===")
    for layer, perf in results['test_performance'].items():
        print(f"{layer}: NSE={perf['NSE']:.3f}, PBIAS={perf['PBIAS']:.1f}%")
    
    # Optimized parameters
    print("\n=== Optimized Parameters ===")
    print(f"Saturated conductivity k_s1: {results['optimized_params']['k_s1']}")
    print(f"Unsaturated conductivity k_u1: {results['optimized_params']['k_u1']}")
    print(f"Capillary rise coefficient k_c1: {results['optimized_params']['k_c1']}")
    
    # Layer information
    if results.get('theta_layers'):
        print("\n=== Layer Configuration ===")
        for layer in results['theta_layers']:
            print(f"Depth {layer['depth']}cm: {layer['input_column']} -> {layer['output_column']}")


def main():
    """Main function - configuration-driven soil moisture model"""
    print("="*60)
    print("Configuration-driven Soil Moisture Model")
    print("="*60)
    
    # 1. Initialize configuration manager
    print("\n1. Initializing configuration...")
    config_manager = ConfigManager()
    
    # 2. Load data
    print("\n2. Loading data...")
    data_2000 = load_data("2000.xlsx", config_manager)
    data_2001 = load_data("2001.xlsx", config_manager)
    
    if data_2000 is None or data_2001 is None:
        print("Error: Data loading failed, program exit")
        sys.exit(1)
    
    # 3. Get layer information
    print("\n3. Analyzing layer configuration...")
    theta_layers = data_2000['theta_layers']
    n_layers = len(theta_layers)
    
    print(f"Detected {n_layers} soil layers:")
    for layer in theta_layers:
        print(f"  - {layer['depth']}cm: {layer['input_column']} -> {layer['output_column']}")
    
    # 4. Initialize model
    print(f"\n4. Initializing {n_layers}-layer soil model...")
    model = SoilModel(n_layers=n_layers)
    
    # 5. Run training and evaluation
    print(f"\n5. Running model training and evaluation...")
    results = train_and_evaluate_model(
        model, 
        data_2000['data'], 
        data_2001['data'], 
        theta_layers
    )
    
    # 6. Print results
    print_results(results)
    
    # 7. Generate visualization
    print(f"\n6. Generating visualization charts...")
    viz_results = generate_all_plots(
        data_2000['data'], 
        data_2001['data'], 
        results,
        config_manager,
        output_prefix='../results/'
    )
    
    # 8. Export evaluation metrics (JSON and visualization files)
    print(f"\n7. Exporting evaluation metrics...")
    try:
        from visualization import export_performance_metrics
        
        metrics_export = export_performance_metrics(
            results.get('train_detailed_performance', {}),
            results.get('test_detailed_performance', {}),
            theta_layers,
            output_prefix='../results/'
        )
    except Exception as e:
        print(f"Warning: Could not export performance metrics: {e}")
        # Fallback to manual summary
        metrics_export = {
            'json_file': None,
            'visualization_file': None,
            'summary': {
                'train_reliable_layers': 0,
                'test_reliable_layers': 0,
                'total_layers': len(theta_layers)
            }
        }
        
        # Calculate summary manually
        if results.get('train_detailed_performance'):
            train_reliable = sum(1 for layer_results in results['train_detailed_performance'].values() 
                               if layer_results.get('overall_status') == 'High Confidence')
            metrics_export['summary']['train_reliable_layers'] = train_reliable
        
        if results.get('test_detailed_performance'):
            test_reliable = sum(1 for layer_results in results['test_detailed_performance'].values() 
                              if layer_results.get('overall_status') == 'High Confidence')
            metrics_export['summary']['test_reliable_layers'] = test_reliable
    
    print(f"\n=== Program Execution Completed ===")
    print(f"Layers used: {n_layers}")
    depth_labels = [f"{layer['depth']}cm" for layer in theta_layers]
    print(f"Layer depths: {depth_labels}")
    print(f"Configuration files: ../config/model_config.txt, ../config/variable_mapping.txt")
    print(f"Generated chart files: {len(viz_results['files_generated'])} files")
    print(f"Evaluation metrics: NSE and PBIAS")
    print(f"Reliable layers: Training set {metrics_export['summary']['train_reliable_layers']}/{metrics_export['summary']['total_layers']}, Test set {metrics_export['summary']['test_reliable_layers']}/{metrics_export['summary']['total_layers']}")
    
    return {
        'model': model,
        'results': results,
        'config': config_manager,
        'layer_info': theta_layers,
        'visualization': viz_results,
        'metrics_export': metrics_export
    }


if __name__ == "__main__":
    try:
        run_results = main()
        print("\nProgram completed successfully!")
    except Exception as e:
        print(f"\nProgram execution error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

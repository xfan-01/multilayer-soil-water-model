#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration Management Module - Flexible parameter and variable mapping management
"""

import os
import pandas as pd
import numpy as np


class ConfigManager:
    """Configuration manager - Flexible parameter and variable mapping management"""
    
    def __init__(self, config_file='../config/model_config.txt', mapping_file='../config/variable_mapping.txt'):
        self.config = {}
        self.mapping = {}
        self.load_config(config_file)
        self.load_mapping(mapping_file)
    
    def load_config(self, config_file):
        """Load model configuration"""
        if not os.path.exists(config_file):
            print(f"Error: Configuration file {config_file} does not exist")
            return
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip()
                            self.config[key] = self._parse_value(value)
            print(f"Successfully loaded configuration file: {config_file}")
        except Exception as e:
            print(f"Error: Failed to load configuration file - {e}")
    
    def load_mapping(self, mapping_file):
        """Load variable mapping"""
        if not os.path.exists(mapping_file):
            print(f"Error: Mapping file {mapping_file} does not exist")
            return
        
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            self.mapping[key.strip()] = value.strip()
            print(f"Successfully loaded variable mapping file: {mapping_file}")
        except Exception as e:
            print(f"Error: Failed to load mapping file - {e}")
    
    def _parse_value(self, value):
        """Parse configuration value type"""
        # Handle null values
        if not value or value.strip() == '':
            return None
            
        value = value.strip()
        
        # Boolean values
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Number lists
        if ',' in value:
            items = [item.strip() for item in value.split(',') if item.strip()]
            if not items:
                return None
            return items
        
        # Single numbers
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        return value
    
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_mapping(self, key, default=None):
        """Get variable mapping"""
        return self.mapping.get(key, default)
    
    def get_theta_mappings(self):
        """Get all theta-related mappings"""
        theta_mappings = {}
        for output_var, input_var in self.mapping.items():
            # Find theta_XXmm format mappings
            if output_var.startswith('theta_') and output_var.endswith('mm'):
                theta_mappings[output_var] = input_var
        return theta_mappings
    
    def get_data_files(self):
        """Get training and test data file configuration"""
        data_dir = self.get_mapping('data_directory', './')
        train_file = self.get_mapping('train_data_file', '2000.xlsx')
        test_file = self.get_mapping('test_data_file', '2001.xlsx')
        
        # Build complete paths
        if not data_dir.endswith('/') and not data_dir.endswith('\\'):
            data_dir += '/'
            
        return {
            'train_file': os.path.join(data_dir, train_file) if data_dir != './' else train_file,
            'test_file': os.path.join(data_dir, test_file) if data_dir != './' else test_file,
            'data_directory': data_dir
        }
    
    def get_unit_conversion_config(self):
        """Get unit conversion configuration"""
        return {
            'precip_conversion': self.get_mapping('precip_unit_conversion', 'false').lower() == 'true',
            'precip_factor': float(self.get_mapping('precip_conversion_factor', '1.0')),
            'theta_conversion': self.get_mapping('theta_unit_conversion', 'true').lower() == 'true',
            'theta_factors': [float(x) for x in self.get_mapping('theta_conversion_factors', '100.0,100.0,100.0').split(',')]
        }
    
    def get_quality_control_config(self):
        """Get data quality control configuration"""
        return {
            'min_precip': float(self.get_mapping('min_precipitation', '0.0')),
            'max_precip': float(self.get_mapping('max_precipitation', '200.0')),
            'min_et': float(self.get_mapping('min_evapotranspiration', '0.0')),
            'max_et': float(self.get_mapping('max_evapotranspiration', '15.0')),
            'missing_values': [x.strip() for x in self.get_mapping('missing_values', 'NaN,NA,null,-999').split(',')]
        }
    
    def validate_data_quality(self, data):
        """Validate data quality according to configuration"""
        qc_config = self.get_quality_control_config()
        
        # Get variable mappings
        precip_col = self.get_mapping('precipitation')
        et_col = self.get_mapping('evapotranspiration')
        
        if precip_col and precip_col in data.columns:
            # Precipitation quality control
            invalid_precip = (data[precip_col] < qc_config['min_precip']) | (data[precip_col] > qc_config['max_precip'])
            if invalid_precip.any():
                print(f"Warning: Found {invalid_precip.sum()} anomalous precipitation values, set to NaN")
                data.loc[invalid_precip, precip_col] = np.nan
        
        if et_col and et_col in data.columns:
            # Evapotranspiration quality control
            invalid_et = (data[et_col] < qc_config['min_et']) | (data[et_col] > qc_config['max_et'])
            if invalid_et.any():
                print(f"Warning: Found {invalid_et.sum()} anomalous evapotranspiration values, set to NaN")
                data.loc[invalid_et, et_col] = np.nan
        
        return data


def load_data(filename, config_manager=None):
    """
    Configuration-driven data loading function
    
    Parameters:
    -----------
    filename : str
        Data file name
    config_manager : ConfigManager, optional
        Configuration manager, creates default configuration if None
        
    Returns:
    --------
    dict : Dictionary containing data and configuration
    """
    if config_manager is None:
        config_manager = ConfigManager()
    
    try:
        print(f"Loading data file: {filename}")
        # Check if file exists in data folder
        data_path = f"../data/{filename}"
        if not os.path.exists(data_path):
            data_path = filename  # Fallback to original path
        data = pd.read_excel(data_path)
        
        # Data quality control
        print("Performing data quality control...")
        data = config_manager.validate_data_quality(data)
        
        # Time processing
        time_col = config_manager.get_mapping('time_column')
        if time_col and time_col in data.columns:
            try:
                data['Date'] = pd.to_datetime(data[time_col], errors='coerce')
                valid_time_mask = data['Date'].notna()
                if valid_time_mask.sum() < len(data):
                    print(f"Warning: Found {(~valid_time_mask).sum()} invalid time values, filtered")
                    data = data[valid_time_mask].copy()
                data.set_index('Date', inplace=True)
            except Exception as e:
                print(f"Time column conversion failed: {e}, using original index")
        
        # Basic statistics
        precip_col = config_manager.get_mapping('precipitation')
        evap_col = config_manager.get_mapping('evapotranspiration')
        
        if precip_col and precip_col in data.columns:
            total_precip = data[precip_col].sum()
            print(f"Total precipitation: {total_precip:.1f} mm")
        if evap_col and evap_col in data.columns:
            total_evap = data[evap_col].sum()
            print(f"Total evaporation: {total_evap:.1f} mm")
        if 'AirT' in data.columns:
            avg_temp = data['AirT'].mean()
            print(f"Annual average temperature: {avg_temp:.1f} °C")
        
        # Variable renaming - according to mapping file
        renamed_data = data.copy()
        
        # Rename meteorological variables
        if precip_col and precip_col in data.columns:
            output_precip = config_manager.get_mapping('output_precip')
            if output_precip:
                renamed_data[output_precip] = data[precip_col]
        
        if evap_col and evap_col in data.columns:
            output_evap = config_manager.get_mapping('output_evap')
            if output_evap:
                renamed_data[output_evap] = data[evap_col]
        
        # Rename theta variables - according to mapping file configuration
        theta_mappings = config_manager.get_theta_mappings()
        theta_layers = []
        
        for output_var, input_var in theta_mappings.items():
            if input_var in data.columns:
                renamed_data[output_var] = data[input_var]
                # Extract depth information (from theta_XXmm format)
                depth = int(output_var.replace('theta_', '').replace('mm', ''))
                theta_layers.append({
                    'depth': depth,
                    'input_column': input_var,
                    'output_column': output_var
                })
                print(f"  Soil layer mapping: {input_var} -> {output_var} ({depth}cm)")
            else:
                print(f"  Warning: Input column '{input_var}' does not exist in data")
        
        # Sort by depth
        theta_layers.sort(key=lambda x: x['depth'])
        
        # Display time range and data summary
        if hasattr(renamed_data.index, 'strftime'):
            start_date = renamed_data.index[0].strftime('%Y-%m-%d')
            end_date = renamed_data.index[-1].strftime('%Y-%m-%d')
            print(f"Data period: {start_date} to {end_date}")
        
        print(f"Total days: {len(renamed_data)} days")
        print("Configured soil layers: " + ", ".join([f"{layer['depth']}cm" for layer in theta_layers]))
        
        return {
            'data': renamed_data,
            'filename': filename,
            'theta_layers': theta_layers,
            'config': config_manager
        }
        
    except Exception as e:
        print(f"Error loading data file {filename}: {e}")
        return None

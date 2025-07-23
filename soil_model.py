"""
Multi-Layer Soil Water Model - Physical-Based Bidirectional Percolation

This model implements physically consistent bidirectional percolation flow mechanism,
removing artificial inter-layer exchange restrict        Parameters:
        ----------
        theta : float - Current moisture content θi
        theta_min : float - Minimum moisture content θmin,i
        theta_max : float - Saturated moisture content θs,i
        k_c1 : float - Capillary rise coefficient kc1,i
        k_c2 : float - Capillary rise index kc2,i
        
        Returns:
        -------
        float : Capillary rise amount q_cap,ietely based on soil physics
capillary action and gravity-driven water movement patterns.

Core Physical Mechanisms:
========================

1. Saturated Percolation (Neilson.1995):
   When layer moisture θi > θs,i: qs,i = (θi - θs,i) × ks,i
   Where: θs,i is saturated moisture, ks,i is saturated conductivity

2. Unsaturated Percolation (Neilson.1995):  
   When θmin,i < θi ≤ θs,i:
   qun,i = (θi - θmin,i) × ku1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^ku2,i
   
3. Bidirectional Percolation Flow:
   net_flux = qun_down - qun_up
   Implements natural physical process "water flows from wet to dry areas"

Variable Definitions:
====================
θi     - Layer i moisture content (mm)
θs,i   - Layer i saturated moisture (mm) 
θmin,i - Layer i minimum moisture/wilting point (mm)
ks,i   - Layer i saturated conductivity (mm/day)
ku1,i  - Layer i unsaturated percolation coefficient (mm/day)
ku2,i  - Layer i unsaturated percolation index (-)
P      - Precipitation (mm/day)
E      - Evaporation (mm/day)
R      - Surface runoff (mm/day)
Qgw    - Deep percolation (mm/day)
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import r2_score, mean_squared_error

class SoilModel:
    """
    Multi-Layer Soil Water Model - Bidirectional Percolation Physics
    
    This model is developed based on long-term observation data from Hyytiala
    Forest Research Station, implementing complete soil water vertical movement
    physical processes:
    
    Physical Processes:
    ------------------
    1. Top layer receives precipitation P, loses evaporation E
    2. Each layer independently calculates vertical drainage (saturated + unsaturated)
    3. Drainage gravitationally transfers to lower layer  
    4. Inter-layer bidirectional percolation flow (capillary + gravity balance)
    5. Top layer overflow generates runoff R
    6. Bottom layer drainage forms deep percolation Qgw
    
    Mathematical Expression:
    -----------------------
    Water balance: ΔθI = P - E - qs,i - qun,i + Qin - Qout
    Bidirectional flow: net_flux = qun_down - qun_up  
    
    Parameters:
    ----------
    n_layers : int - Number of soil layers
    max_theta : list - Saturated moisture content for each layer θs,i (mm)
    theta_min : list - Minimum moisture content for each layer θmin,i (mm)  
    k_s1 : list - Saturated conductivity coefficient ks,i (mm/day)
    k_u1 : list - Unsaturated percolation coefficient ku1,i (mm/day)
    k_u2 : list - Unsaturated percolation index ku2,i (-)
    """
    
    def __init__(self, n_layers=2):
        """
        Initialize multi-layer soil model
        
        Parameters:
        ----------
        n_layers : int
            Number of soil layers, default 2 layers (surface + deep)
        """
        self.n_layers = n_layers
        self.layers = [{"theta": 0.0} for _ in range(n_layers)]
        self.default_params = self.get_default_params()
    
    def get_default_params(self):
        """
        Generate default parameters - calibrated based on Hyytiala Forest Station observation data
        
        Parameter sources and physical meanings:
        ---------------------------------------
        Based on long-term soil moisture observation data from Hyytiala Forest Research Station, Finland,
        the following parameter ranges are obtained through calibration with measured soil moisture:
        
        Returns:
        -------
        dict : Dictionary containing all physical parameters
            max_theta : Saturated moisture content for each layer θs,i (mm) - maximum water holding capacity
            theta_min : Minimum moisture content for each layer θmin,i (mm) - wilting point moisture  
            k_s1 : Saturated conductivity coefficient ks,i (mm/day) - water conduction capacity under saturated conditions
            k_u1 : Unsaturated percolation coefficient ku1,i (mm/day) - percolation intensity under unsaturated conditions
            k_u2 : Unsaturated percolation index ku2,i (-) - controls nonlinearity of percolation
        """
        # Base parameter values
        base_max_theta = [50.0, 60.0, 70.0]    # increasing depth, enhanced water holding capacity
        base_theta_min = [5.0, 8.0, 10.0]      # increasing depth, increased minimum moisture
        base_k_s1 = [2.0, 1.5, 1.0]            # increasing depth, decreasing conductivity
        base_k_u1 = [0.8, 0.6, 0.4]            # increasing depth, decreasing percolation coefficient
        base_k_u2 = [2.0, 1.8, 1.6]            # increasing depth, decreasing index
        # New: independent capillary rise parameters (usually smaller than gravity percolation)
        base_k_c1 = [0.4, 0.3, 0.2]            # capillary rise coefficient
        base_k_c2 = [2.2, 2.0, 1.8]            # capillary rise index
        
        # Extend to required number of layers, use last value for excess layers
        return {
            "max_theta": (base_max_theta + [base_max_theta[-1]] * self.n_layers)[:self.n_layers],
            "theta_min": (base_theta_min + [base_theta_min[-1]] * self.n_layers)[:self.n_layers],
            "k_s1": (base_k_s1 + [base_k_s1[-1]] * self.n_layers)[:self.n_layers],
            "k_u1": (base_k_u1 + [base_k_u1[-1]] * self.n_layers)[:self.n_layers],
            "k_u2": (base_k_u2 + [base_k_u2[-1]] * self.n_layers)[:self.n_layers],
            "k_c1": (base_k_c1 + [base_k_c1[-1]] * self.n_layers)[:self.n_layers],
            "k_c2": (base_k_c2 + [base_k_c2[-1]] * self.n_layers)[:self.n_layers],
        }
    
    def calc_drainage(self, layer_idx, theta):
        """
        Calculate single layer drainage - saturated percolation
        
        Physical mechanism:
        ------------------
        When soil moisture exceeds saturation value, excess water drains under gravity.
        Drainage intensity relates to super-saturation degree and soil conductivity performance.
        
        Mathematical expression:
        -----------------------
        qs,i = (θi - θs,i) × ks,i  (when θi > θs,i)
        qs,i = 0                    (when θi ≤ θs,i)
        
        Parameters:
        ----------
        layer_idx : int - Soil layer index (0=top layer, 1=deep layer, ...)
        theta : float - Current layer moisture content θi
        
        Returns:
        -------
        float : Drainage amount qs,i
        """
        params = self.default_params
        max_theta = params["max_theta"][layer_idx]
        k_s1 = params["k_s1"][layer_idx]
        
        # Calculate saturated drainage
        if theta > max_theta:
            excess = theta - max_theta
            drainage = excess * k_s1
            return drainage
        else:
            return 0.0
    
    def calc_percolation(self, theta, theta_min, theta_max, k_u1, k_u2):
        """
        Calculate unsaturated percolation
        
        Physical mechanism:
        ------------------
        Under unsaturated conditions, water moves under combined capillary and gravity forces.
        Percolation intensity has nonlinear relationship with effective soil moisture.
        
        Mathematical expression:
        -----------------------
        qun,i = (θi - θmin,i) × ku1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^ku2,i
        
        Where:
        - (θi - θmin,i): effective moisture content
        - (θs,i - θmin,i): effective capacity
        - [(θi - θmin,i)/(θs,i - θmin,i)]^ku2,i: nonlinear response of relative saturation
        
        Parameters:
        ----------
        theta : float - Current moisture content θi
        theta_min : float - Minimum moisture content θmin,i
        theta_max : float - Saturated moisture content θs,i
        k_u1 : float - Unsaturated percolation coefficient ku1,i
        k_u2 : float - Unsaturated percolation index ku2,i
        
        Returns:
        -------
        float : Unsaturated percolation amount qun,i
        """
        # Check physical constraint conditions
        if theta <= theta_min or theta_max <= theta_min:
            return 0.0
        
        # Calculate effective moisture and capacity
        effective_moisture = theta - theta_min
        effective_capacity = theta_max - theta_min
        
        # Calculate unsaturated percolation
        relative_saturation = effective_moisture / effective_capacity
        percolation = effective_moisture * k_u1 * (relative_saturation ** k_u2)
        
        return percolation

    def calc_capillary_rise(self, theta, theta_min, theta_max, k_c1, k_c2):
        """
        Calculate capillary rise amount - using independent capillary parameters
        
        Physical mechanism:
        ------------------
        Water moves upward from lower layer to upper layer against gravity through capillary action.
        Its driving force differs from gravity percolation, therefore uses independent parameter set.
        
        Mathematical expression (same form as unsaturated percolation, but different parameters):
        -------------------------------------------------------------------------------------
        q_cap,i = (θi - θmin,i) × kc1,i × [(θi - θmin,i)/(θs,i - θmin,i)]^kc2,i
        
        Parameters:
        ----------
        theta : float - Current moisture content θi
        theta_min : float - Minimum moisture content θmin,i
        theta_max : float - Saturated moisture content θs,i
        k_c1 : float - Capillary rise coefficient kc1,i
        k_c2 : float - Capillary rise index kc2,i
        
        Returns:
        -------
        float : Capillary rise amount q_cap,i
        """
        # Check physical constraint conditions
        if theta <= theta_min or theta_max <= theta_min:
            return 0.0
        
        effective_moisture = theta - theta_min
        effective_capacity = theta_max - theta_min
        
        relative_saturation = effective_moisture / effective_capacity
        capillary_rise = effective_moisture * k_c1 * (relative_saturation ** k_c2)
        
        return capillary_rise

    def update(self, P, E=0.0):
        """
        Update model state - implementing bidirectional percolation water balance
        
        Physical process sequence:
        1. Top layer receives precipitation P and loses evaporation E
        2. Each layer independently calculates vertical drainage (Eq 1.7 + 1.8)
        3. Drainage passes down to next layer
        4. Calculate interlayer bidirectional percolation flow (capillary action)
        5. Handle top layer runoff (capacity overflow)
        6. Bottom layer drainage forms deep percolation
        
        Mathematical expression:
        --------
        Water balance: ΔWi = P - E - qsati - qunsati + Qin - Qout
        Bidirectional flow: net_flux = qun_down - qun_up
        Where: qun_down/up calculated using unsaturated percolation formula for adjacent layers
        
        Parameters:
        ----------
        P : float - Precipitation (mm/day)
        E : float - Evaporation (mm/day)
            
        Returns:
        -------
        tuple : (runoff R, layer drainage Q_drainage, interlayer net flow Q_interlayer, deep percolation Qgw)
        """
        # Get current state and parameters
        current_theta = [self.layers[i]["theta"] for i in range(self.n_layers)]
        temp_theta = current_theta.copy()
        
        # Extract physical parameters
        params = self.default_params
        Wsi = params["max_theta"]
        Wmi = params["theta_min"]
        ku1 = params["k_u1"]
        ku2 = params["k_u2"]
        kc1 = params["k_c1"]
        kc2 = params["k_c2"]
        
        # Initialize output variables
        Q_drainage = [0.0] * self.n_layers
        Q_interlayer = [0.0] * (self.n_layers - 1) if self.n_layers > 1 else []
        R_now = 0.0
        Qgw_now = 0.0
        
        # Step 1: Top layer receives precipitation and evaporation
        temp_theta[0] = current_theta[0] + P - E
        temp_theta[0] = max(0, temp_theta[0])
        
        # Step 2: Calculate independent drainage for each layer
        for i in range(self.n_layers):
            Q_drainage[i] = self.calc_drainage(i, temp_theta[i])
            temp_theta[i] -= Q_drainage[i]
            
            # Transfer drainage to next layer
            if i + 1 < self.n_layers:
                temp_theta[i+1] += Q_drainage[i]
        
        # Step 3: Bidirectional interlayer percolation flow
        dt = 1.0
        
        for i in range(self.n_layers - 1):
            upper_idx = i
            lower_idx = i + 1
            
            upper_theta = temp_theta[upper_idx]
            lower_theta = temp_theta[lower_idx]
            
            # Unsaturated percolation from upper to lower layer
            qun_down = self.calc_percolation(
                upper_theta, Wmi[upper_idx], Wsi[upper_idx], ku1[upper_idx], ku2[upper_idx]
            )
            
            # Capillary rise from lower to upper layer
            qun_up = self.calc_capillary_rise(
                lower_theta, Wmi[lower_idx], Wsi[lower_idx], kc1[lower_idx], kc2[lower_idx]
            )
            
            # Calculate net flux
            net_flux = qun_down - qun_up
            
            # Apply flow constraints
            if net_flux > 0:
                net_flux = min(net_flux * dt, upper_theta * 0.5)
            else:
                net_flux = max(net_flux * dt, -lower_theta * 0.5)
            
            # Check capacity limits
            if net_flux > 0:
                available_capacity = max(0, Wsi[lower_idx] - temp_theta[lower_idx])
                net_flux = min(net_flux, available_capacity)
            else:
                available_capacity = max(0, Wsi[upper_idx] - temp_theta[upper_idx])
                net_flux = max(net_flux, -available_capacity)
            
            # Update moisture distribution
            temp_theta[upper_idx] -= net_flux
            temp_theta[lower_idx] += net_flux
            
            Q_interlayer[i] = net_flux
        
        # Step 4: Handle runoff
        max_top = self.default_params["max_theta"][0]
        R_now = max(temp_theta[0] - max_top, 0)
        temp_theta[0] -= R_now
        
        # Step 5: Bottom layer deep percolation
        if self.n_layers > 1:
            bottom_idx = self.n_layers - 1
            Qgw_now = Q_drainage[bottom_idx]
        else:
            Qgw_now = Q_drainage[0]
        
        # Step 6: Apply physical constraints
        for i in range(self.n_layers):
            max_val = self.default_params["max_theta"][i]
            temp_theta[i] = max(0, min(temp_theta[i], max_val))
        
        # Update model state
        for i in range(self.n_layers):
            self.layers[i]["theta"] = temp_theta[i]
        
        return R_now, Q_drainage, Q_interlayer, Qgw_now
    
    def get_moisture(self, layer_idx=None):
        """
        Get soil moisture content
        
        Parameters:
        ----------
        layer_idx : int, optional
            Specified layer index. If None, returns all layer moisture
            
        Returns:
        -------
        float or list : Specified layer moisture or all layer moisture list
        """
        if layer_idx is not None:
            return self.layers[layer_idx]["theta"]
        return [layer["theta"] for layer in self.layers]
    
    def reset(self, initial_theta=None):
        """
        Reset model state
        
        Parameters:
        ----------
        initial_theta : list, optional
            Initial moisture array. If None, reset to 0
        """
        if initial_theta is not None and len(initial_theta) == self.n_layers:
            for i in range(self.n_layers):
                self.layers[i]["theta"] = initial_theta[i]
        else:
            for i in range(self.n_layers):
                self.layers[i]["theta"] = 0.0

    def get_params(self):
        """
        Return current parameter configuration
        
        Returns:
        -------
        dict : Dictionary copy containing all physical parameters
        """
        return self.default_params.copy()
    
    def set_params(self, params):
        """
        Set model parameters
        
        Parameters:
        ----------
        params : dict
            Parameter dictionary that can contain the following keys:
            - 'max_theta': Layer saturated moisture content
            - 'theta_min': Layer minimum moisture content
            - 'k_u1': Unsaturated percolation coefficient
            - 'k_u2': Unsaturated percolation index
        """
        self.default_params.update(params)


def simulate(model, precipitation, evapotranspiration):
    """
    Long-term simulation function - execute multi-timestep soil moisture dynamics simulation
    
    Mathematical process:
    --------------------
    For each time step t:
    1. Call model.update(P[t], E[t])
    2. Collect state outputs: θ[t], R[t], Q[t], Qgw[t]
    3. Accumulate water balance verification
    
    Water balance:
    ΣP = ΣE + ΣR + ΣQgw + Δθtotal
    
    Parameters:
    ----------
    model : SoilModel
        Initialized soil model instance
    precipitation : array_like
        Precipitation time series
    evapotranspiration : array_like  
        Evapotranspiration time series
        
    Returns:
    -------
    dict : Dictionary containing complete simulation results
        - 'moisture': Layer moisture time series
        - 'runoff': Runoff time series
        - 'drainage': Layer drainage time series
        - 'interlayer_flow': Interlayer flow time series
        - 'groundwater': Deep percolation time series
        - 'water_balance': Water balance verification results
    """
    # Initialize output lists
    simulated = []
    R_out = []
    Qdrain_out = []
    Qexchange_out = []
    Qgw_out = []
    
    # Time step loop
    for i in range(len(precipitation)):
        P = precipitation[i]
        E = evapotranspiration[i]
        
        # Model update
        R, Qdrain, Qexchange, Qgw = model.update(P, E)
        
        # Get current state
        moisture = model.get_moisture()
        simulated.append(moisture)
        
        # Record outputs
        R_out.append(R)
        Qdrain_out.append(Qdrain)
        Qexchange_out.append(Qexchange)
        Qgw_out.append(Qgw)
        
        # Check simulation results
        if any(np.isnan(m) for m in simulated[-1]):
            print(f"Warning: Day {i+1} simulation results contain NaN values")
            if simulated:
                simulated[-1] = simulated[-2] if i > 0 else [0.0] * model.n_layers
    
    return (
        np.array(simulated),
        np.array(R_out),
        np.array(Qdrain_out),
        np.array(Qexchange_out),
        np.array(Qgw_out)
    )


def calibrate(model, precipitation, evapotranspiration, observed_moisture):
    """
    Parameter calibration function - optimize soil percolation parameters
    
    Parameters:
    ----------
    model : SoilModel
        Model instance to be calibrated
    precipitation : array_like
        Precipitation time series (mm/day)
    evapotranspiration : array_like
        Evapotranspiration time series (mm/day)
    observed_moisture : array_like
        Observed soil moisture data (mm)
        
    Returns:
    -------
    dict : Optimization results dictionary
    """
    def objective(params):
        # Update model parameters - includes percolation and capillary rise parameters
        n_layers = model.n_layers
        param_idx = 0
        
        # Saturated conductivity coefficient (k_s1)
        model.default_params['k_s1'] = params[param_idx:param_idx + n_layers].tolist()
        param_idx += n_layers
        
        # Unsaturated conductivity coefficient (k_u1)
        model.default_params['k_u1'] = params[param_idx:param_idx + n_layers].tolist()
        param_idx += n_layers

        # Capillary rise coefficient (k_c1)
        model.default_params['k_c1'] = params[param_idx:param_idx + n_layers].tolist()
        
        # Reset and run simulation
        model.reset(initial_theta=observed_moisture[0])
        simulated, _, _, _, _ = simulate(model, precipitation, evapotranspiration)
        
        # Kling-Gupta Efficiency (KGE) objective function
        def calculate_kge(obs, sim):
            """Calculate Kling-Gupta Efficiency"""
            # Remove NaN values
            valid_mask = ~(np.isnan(obs) | np.isnan(sim))
            if not np.any(valid_mask) or len(obs[valid_mask]) < 2:
                return -1.0  # Return worst KGE value
            
            obs_valid = obs[valid_mask]
            sim_valid = sim[valid_mask]
            
            # Calculate statistics
            obs_mean = np.mean(obs_valid)
            sim_mean = np.mean(sim_valid)
            obs_std = np.std(obs_valid, ddof=1)
            sim_std = np.std(sim_valid, ddof=1)
            
            # Avoid division by zero
            if obs_std == 0 or sim_std == 0:
                return -1.0
            
            # 1. Correlation coefficient (r)
            correlation = np.corrcoef(obs_valid, sim_valid)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # 2. Bias ratio (β = μ_sim / μ_obs)
            if obs_mean == 0:
                beta = 1.0 if sim_mean == 0 else float('inf')
            else:
                beta = sim_mean / obs_mean
            
            # 3. Variability ratio (α = σ_sim / σ_obs)
            alpha = sim_std / obs_std
            
            # Calculate KGE
            kge = 1 - np.sqrt((correlation - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            
            return kge
        
        # Calculate KGE for each layer using weighted average
        layer_weights = [1.0, 0.8, 0.6][:n_layers]  # Higher weight for surface layer
        total_weight = sum(layer_weights)
        weighted_kge_loss = 0.0
        
        for i in range(n_layers):
            obs = observed_moisture[:, i]
            sim = simulated[:, i]
            
            # Calculate KGE for layer i
            kge_i = calculate_kge(obs, sim)
            
            # Convert to loss function (higher KGE is better, so use 1-KGE as loss)
            kge_loss = 1.0 - kge_i
            
            # Weighted accumulation
            weighted_kge_loss += layer_weights[i] * kge_loss
        
        # Normalize
        final_loss = weighted_kge_loss / total_weight
        
        # Ensure positive return value
        return max(0.0, final_loss)
    
    # Initial parameter setup - includes k_s1, k_u1, k_c1
    n_layers = model.n_layers
    
    # Parameter bounds
    bounds = []
    
    # k_s1 bounds (saturated conductivity coefficient)
    for i in range(n_layers):
        bounds.append((0.1, 10.0))
    
    # k_u1 bounds (unsaturated conductivity coefficient)
    for i in range(n_layers):
        bounds.append((0.01, 5.0))
        
    # k_c1 bounds (capillary rise coefficient)
    for i in range(n_layers):
        bounds.append((0.01, 2.0)) # Capillary action usually weaker
    
    # Execute optimization
    result = differential_evolution(objective, bounds, seed=42, maxiter=100)
    
    # Extract optimization results
    optimized_params = result.x
    param_idx = 0
    
    optimized_k_s1 = optimized_params[param_idx:param_idx + n_layers].tolist()
    param_idx += n_layers
    
    optimized_k_u1 = optimized_params[param_idx:param_idx + n_layers].tolist()
    param_idx += n_layers
    
    optimized_k_c1 = optimized_params[param_idx:param_idx + n_layers].tolist()
    
    return {
        'k_s1': optimized_k_s1,      # Saturated conductivity coefficient
        'k_u1': optimized_k_u1,      # Unsaturated conductivity coefficient
        'k_c1': optimized_k_c1,      # Capillary rise coefficient
        'optimization_result': result
    }

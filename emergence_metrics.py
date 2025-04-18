"""
Emergence Metrics Implementation based on ReconcilingEmergences framework

This module provides functions to calculate causal emergence metrics:
- Delta: Downward causation criterion (macro → micro effect)
- Gamma: Causal decoupling criterion (comparison of macro vs micro predictive power)
- Psi: Causal emergence criterion (information gain from macro)

Reference:
    Rosas*, Mediano*, et al. (2020). Reconciling emergences: An
    information-theoretic approach to identify causal emergence in
    multivariate data. https://arxiv.org/abs/2004.08220
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, Union

def emergence_delta(X: np.ndarray, V: np.ndarray, 
                    tau: int = 1, 
                    mutual_info_func=None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute downward causation criterion (delta) from data.
    
    Parameters:
    -----------
    X : np.ndarray
        Micro time series, shape (T, D) where T is time steps, D is micro dimensions
    V : np.ndarray
        Macro time series, shape (T, R) where T is time steps, R is macro dimensions
    tau : int, optional
        Time delay for calculating mutual information, by default 1
    mutual_info_func : callable, optional
        Function to compute mutual information, by default None
        If None, mutual_info from info_theory_utils is used
        
    Returns:
    --------
    delta : float
        Downward causation criterion: max(v_mi - x_mi)
    v_mi : np.ndarray
        Mutual information between macro variables at t and each micro variable at t+tau
    x_mi : np.ndarray
        Sum of mutual information between each micro variable at t and each micro variable at t+tau
    """
    if mutual_info_func is None:
        try:
            # Try to import mutual_info from info_theory_utils
            import info_theory_utils
            mutual_info_func = info_theory_utils.mutual_info
        except (ImportError, AttributeError):
            raise ImportError("mutual_info function not found in info_theory_utils. "
                             "Please provide a mutual_info_func.")
    
    # Check dimensions
    if X.shape[0] != V.shape[0]:
        raise ValueError("X and V must have the same number of time steps")
    
    # Split data into past and future
    X_past = X[:-tau]
    X_future = X[tau:]
    V_past = V[:-tau]
    
    print(f"emergence_delta - データ形状: X_past: {X_past.shape}, X_future: {X_future.shape}, V_past: {V_past.shape}")
    
    # Calculate mutual information for each target micro variable
    n_micro_vars = X.shape[1]
    v_mi = np.zeros(n_micro_vars)
    x_mi = np.zeros(n_micro_vars)
    
    # Loop through each target micro variable
    for j in range(n_micro_vars):
        # MI between past macro and future micro j
        target_future = X_future[:, j:j+1]
        v_mi[j] = mutual_info_func(V_past, target_future)
        
        # Sum of MI between past micro variables and future micro j
        micro_mi_sum = 0
        for i in range(n_micro_vars):
            source_past = X_past[:, i:i+1]
            mi_val = mutual_info_func(source_past, target_future)
            micro_mi_sum += mi_val
        x_mi[j] = micro_mi_sum
    
    # Calculate delta as max(v_mi - x_mi)
    delta = np.max(v_mi - x_mi)
    
    return delta, v_mi, x_mi

def emergence_gamma(X: np.ndarray, V: np.ndarray, 
                   tau: int = 1,
                   mutual_info_func=None) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute causal decoupling criterion (gamma) from data.
    
    Parameters:
    -----------
    X : np.ndarray
        Micro time series, shape (T, D) where T is time steps, D is micro dimensions
    V : np.ndarray
        Macro time series, shape (T, R) where T is time steps, R is macro dimensions
    tau : int, optional
        Time delay for calculating mutual information, by default 1
    mutual_info_func : callable, optional
        Function to compute mutual information, by default None
        
    Returns:
    --------
    gamma : float
        Causal decoupling criterion: max(v_mi - s_mi)
    v_mi : np.ndarray
        Mutual information between macro variables at t and each micro variable at t+tau
    s_mi : np.ndarray
        Mutual information between each micro variable at t and each micro variable at t+tau
    """
    if mutual_info_func is None:
        try:
            import info_theory_utils
            mutual_info_func = info_theory_utils.mutual_info
        except (ImportError, AttributeError):
            raise ImportError("mutual_info function not found in info_theory_utils")
    
    # Check dimensions
    if X.shape[0] != V.shape[0]:
        raise ValueError("X and V must have the same number of time steps")
    
    # Split data into past and future
    X_past = X[:-tau]
    X_future = X[tau:]
    V_past = V[:-tau]
    
    n_micro_vars = X.shape[1]
    v_mi = np.zeros(n_micro_vars)
    s_mi = np.zeros(n_micro_vars)
    
    for j in range(n_micro_vars):
        # Prepare target variable
        target_future = X_future[:, j:j+1]
        
        # MI between past macro and future micro j
        v_mi[j] = mutual_info_func(V_past, target_future)
        
        # MI between past micro j and future micro j (self-prediction)
        source_past = X_past[:, j:j+1]
        s_mi[j] = mutual_info_func(source_past, target_future)
    
    # Calculate gamma as max(v_mi - s_mi)
    gamma = np.max(v_mi - s_mi)
    
    return gamma, v_mi, s_mi

def emergence_psi(X: np.ndarray, V: np.ndarray,
                 tau: int = 1,
                 mutual_info_func=None,
                 cond_mutual_info_func=None) -> float:
    """
    Compute causal emergence criterion (psi) from data.
    
    Parameters:
    -----------
    X : np.ndarray
        Micro time series, shape (T, D) where T is time steps, D is micro dimensions
    V : np.ndarray
        Macro time series, shape (T, R) where T is time steps, R is macro dimensions
    tau : int, optional
        Time delay, by default 1
    mutual_info_func : callable, optional
        Function to compute mutual information, by default None
    cond_mutual_info_func : callable, optional
        Function to compute conditional mutual information, by default None
        
    Returns:
    --------
    psi : float
        Causal emergence criterion: I(V_t; X_{t+tau} | X_t)
    """
    if mutual_info_func is None or cond_mutual_info_func is None:
        try:
            import info_theory_utils
            if mutual_info_func is None:
                mutual_info_func = info_theory_utils.mutual_info
            if cond_mutual_info_func is None:
                cond_mutual_info_func = info_theory_utils.conditional_mutual_info
        except (ImportError, AttributeError):
            raise ImportError("Required information theory functions not found")
    
    # Check dimensions
    if X.shape[0] != V.shape[0]:
        raise ValueError("X and V must have the same number of time steps")
    
    # Split data into past and future
    X_past = X[:-tau]
    X_future = X[tau:]
    V_past = V[:-tau]
    
    # Calculate conditional mutual information: I(V_t; X_{t+tau} | X_t)
    # Correct approach: calculate CMI directly between V_past, X_future, conditioned on X_past
    # This respects the joint probability distribution of the full micro state
    psi = cond_mutual_info_func(V_past, X_future, X_past)
    
    return psi

def calculate_all_emergence_metrics(X: np.ndarray, V: np.ndarray, 
                                   tau: int = 1,
                                   mutual_info_func=None,
                                   cond_mutual_info_func=None) -> Dict[str, float]:
    """
    Calculate all emergence metrics (Delta, Gamma, Psi) from data.
    
    Parameters:
    -----------
    X : np.ndarray
        Micro time series, shape (T, D)
    V : np.ndarray
        Macro time series, shape (T, R)
    tau : int, optional
        Time delay, by default 1
    mutual_info_func : callable, optional
        Function to compute mutual information, by default None
    cond_mutual_info_func : callable, optional
        Function to compute conditional mutual information, by default None
        
    Returns:
    --------
    metrics : dict
        Dictionary containing Delta, Gamma, and Psi values
    """
    metrics = {}
    
    # Calculate Delta
    try:
        delta, _, _ = emergence_delta(X, V, tau, mutual_info_func)
        metrics['Delta'] = delta
    except Exception as e:
        print(f"Error calculating Delta: {e}")
        metrics['Delta'] = 0.0
    
    # Calculate Gamma
    try:
        gamma, _, _ = emergence_gamma(X, V, tau, mutual_info_func)
        metrics['Gamma'] = gamma
    except Exception as e:
        print(f"Error calculating Gamma: {e}")
        metrics['Gamma'] = 0.0
    
    # Calculate Psi
    try:
        psi = emergence_psi(X, V, tau, mutual_info_func, cond_mutual_info_func)
        metrics['Psi'] = psi
    except Exception as e:
        print(f"Error calculating Psi: {e}")
        metrics['Psi'] = 0.0
    
    return metrics

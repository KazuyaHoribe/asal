#!/usr/bin/env python
"""
Information Theory Utilities for Causal Blanket Analysis

This module provides functions to compute information-theoretic measures
such as entropy, mutual information, conditional mutual information,
and partial information decomposition.
"""

# Make sure traceback is the first import
import traceback
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression

def validate_and_convert(data, preserve_structure=False):
    """
    Validate input data and convert to numpy array with proper formatting.
    Handles lists, arrays with different shapes, and ensures 1D/2D arrays.
    
    Args:
        data: Input data to validate and convert
        preserve_structure: If True, keeps original structure for multidimensional arrays
    """
    if isinstance(data, list):
        # Try to convert list to numpy array
        try:
            # For nested lists with different lengths, use dtype=object
            data = np.array(data, dtype=object)
            # If it's a 1D array of objects, try to stack them
            if data.ndim == 1 and len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
                # Check if all elements have the same length
                lengths = [len(x) for x in data]
                if len(set(lengths)) == 1:
                    # All elements have the same length, stack them
                    data = np.vstack(data)
                else:
                    # Elements have different lengths, keep as object array
                    pass
        except ValueError:
            # If conversion fails, keep as list
            pass

    # If data is now a numpy array
    if isinstance(data, np.ndarray):
        # Ensure data is at least 1D
        if data.ndim == 0:
            data = np.array([data])
        elif data.ndim > 2 and not preserve_structure:
            # Store original shape before reshaping
            orig_shape = data.shape
            print(f"Flattening data from shape {orig_shape} to 2D")
            # Reshape multi-dimensional arrays to 2D while preserving first dimension
            data = data.reshape(orig_shape[0], -1)
    
    return data

def hist_1d(data, bins=10, range=None):
    """
    Return normalized 1D histogram as estimated probability distribution.
    
    Args:
        data: 1D array with shape (N,)
        bins: Number of bins
        range: Optional tuple of (min, max)
        
    Returns:
        pdf: Normalized probability distribution
        edges: Bin edges
    """
    hist, edges = np.histogram(data, bins=bins, range=range, density=False)
    # Normalize by dividing by total count
    pdf = hist.astype(np.float64) / np.sum(hist)
    return pdf, edges

def hist_2d(xdata, ydata, bins=10, range=None):
    """
    Compute a 2D histogram of two datasets with robust handling of different input types.
    
    Args:
        xdata: First dataset
        ydata: Second dataset
        bins: Number of bins or bin edges
        range: Range for binning
        
    Returns:
        hist: 2D histogram
        xedges: Bin edges for x-axis
        yedges: Bin edges for y-axis
    """
    try:
        # Convert and validate inputs
        xdata = validate_and_convert(xdata)
        ydata = validate_and_convert(ydata)
        
        # Additional validation to ensure we have valid arrays
        if xdata is None or ydata is None:
            print("Warning: xdata or ydata is None")
            raise ValueError("Input data is None")
            
        if np.size(xdata) == 0 or np.size(ydata) == 0:
            print("Warning: xdata or ydata is empty")
            raise ValueError("Empty input data")
        
        # Make sure both are 1D arrays for histogram2d
        if xdata.ndim > 1:
            if xdata.shape[1] == 1:
                xdata = xdata.flatten()
            else:
                # For multi-column x data, use the first column
                print(f"Warning: xdata has shape {xdata.shape}, using first column for histogram")
                xdata = xdata[:, 0]
        
        if ydata.ndim > 1:
            if ydata.shape[1] == 1:
                ydata = ydata.flatten()
            else:
                # For multi-column y data, use the first column
                print(f"Warning: ydata has shape {ydata.shape}, using first column for histogram")
                ydata = ydata[:, 0]
        
        # Make sure both arrays have the same length
        min_length = min(len(xdata), len(ydata))
        if len(xdata) != len(ydata):
            print(f"Warning: xdata and ydata have different lengths ({len(xdata)} vs {len(ydata)}), truncating to {min_length}")
            xdata = xdata[:min_length]
            ydata = ydata[:min_length]
        
        # Additional check after preprocessing
        if min_length == 0:
            print("Warning: data length became 0 after preprocessing")
            raise ValueError("Empty processed data")
        
        # Calculate histogram
        hist, xedges, yedges = np.histogram2d(xdata, ydata, bins=bins, range=range)
        return hist, xedges, yedges
    
    except Exception as e:
        print(f"Error in hist_2d: {e}")
        print(f"xdata shape: {getattr(xdata, 'shape', 'unknown')}, type: {type(xdata)}")
        print(f"ydata shape: {getattr(ydata, 'shape', 'unknown')}, type: {type(ydata)}")
        # Return fallback values
        try:
            x_min = np.min(xdata) if np.size(xdata) > 0 else 0
            x_max = np.max(xdata) if np.size(xdata) > 0 else 1
            y_min = np.min(ydata) if np.size(ydata) > 0 else 0
            y_max = np.max(ydata) if np.size(ydata) > 0 else 1
            
            x_bins = np.linspace(x_min, x_max, bins+1)
            y_bins = np.linspace(y_min, y_max, bins+1)
        except Exception:
            # If even that fails, use default range
            x_bins = np.linspace(0, 1, bins+1)
            y_bins = np.linspace(0, 1, bins+1)
            
        return np.zeros((bins, bins)), x_bins, y_bins

def entropy(data, bins=10):
    """
    Calculate the Shannon entropy of a dataset.
    
    Args:
        data: Input data
        bins: Number of bins for histogram
        
    Returns:
        entropy: Shannon entropy in bits
    """
    data = validate_and_convert(data)
    
    if data.ndim > 1:
        # For multi-column data, we'll compute the joint entropy
        # by creating a histogram of the data
        hist, _ = np.histogramdd(data, bins=bins)
    else:
        # For 1D data, use a simple histogram
        hist, _ = np.histogram(data, bins=bins)
    
    # Normalize to get probability distribution
    probs = hist / np.sum(hist)
    
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    
    # Compute entropy
    return -np.sum(probs * np.log2(probs))

def mutual_information(xdata, ydata, bins=10, hist_range=None, preserve_structure=False):
    """
    Calculate the mutual information between two datasets.
    
    Args:
        xdata: First dataset
        ydata: Second dataset
        bins: Number of bins for histogram
        hist_range: Range for binning
        preserve_structure: If True, preserve multidimensional structure (default: False)
        
    Returns:
        mi: Mutual information in bits
    """
    try:
        # Validate inputs are not None
        if xdata is None or ydata is None:
            print("Warning: Input data is None")
            return 0.0
            
        # Pass preserve_structure flag to validate_and_convert
        xdata = validate_and_convert(xdata, preserve_structure=preserve_structure)
        ydata = validate_and_convert(ydata, preserve_structure=preserve_structure)
        
        # Additional validation after conversion
        if xdata is None or ydata is None or np.size(xdata) == 0 or np.size(ydata) == 0:
            print("Warning: Converted data is None or empty")
            return 0.0
        
        print(f"Debug - mutual_information input shapes: xdata {xdata.shape}, ydata {ydata.shape}")
        
        # If preserving structure but shapes don't match for histogram calculation,
        # we need to flatten the data for processing
        if preserve_structure and (xdata.ndim > 2 or ydata.ndim > 2):
            print(f"Converting preserved structures to 2D for histogram calculation")
            xdata_flat = xdata.reshape(xdata.shape[0], -1)
            ydata_flat = ydata.reshape(ydata.shape[0], -1)
            
            # Use the first column for calculation if the shapes differ after flattening
            if xdata_flat.shape[1] != ydata_flat.shape[1]:
                print(f"Warning: Flattened shapes don't match, using first dimension only")
                xdata_calc = xdata_flat[:, 0]
                ydata_calc = ydata_flat[:, 0]
            else:
                # Use full flattened data
                xdata_calc = xdata_flat
                ydata_calc = ydata_flat
        else:
            xdata_calc = xdata
            ydata_calc = ydata
        
        # Calculate joint histogram
        pxy, xedges, yedges = hist_2d(xdata_calc, ydata_calc, bins=bins, range=hist_range)
        
        # Ensure pxy is a valid array with non-zero sum
        if pxy is None:
            print("Warning: Joint histogram returned None")
            return 0.0
            
        pxy_sum = np.sum(pxy)
        if pxy_sum == 0:
            print("Warning: Joint histogram sum is zero")
            return 0.0
        
        # Calculate marginal histograms
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # Check for valid distributions
        if px is None or py is None:
            print("Warning: Marginal distribution is None")
            return 0.0
            
        px_sum = np.sum(px)
        py_sum = np.sum(py)
        
        if px_sum <= 0 or py_sum <= 0:
            print(f"Warning: Marginal distribution has invalid sum: px_sum={px_sum}, py_sum={py_sum}")
            return 0.0
        
        # Normalize to get probability distributions
        try:
            pxy_normalized = pxy / pxy_sum
            px_normalized = px / px_sum
            py_normalized = py / py_sum
            
            # Validate normalized distributions
            if np.any(np.isnan(pxy_normalized)) or np.any(np.isnan(px_normalized)) or np.any(np.isnan(py_normalized)):
                print("Warning: NaN values in normalized distributions")
                return 0.0
                
        except Exception as e:
            print(f"Warning: Error during normalization: {e}")
            return 0.0
        
        # Calculate mutual information - using direct array shapes to avoid range() issues
        mi = 0.0
        try:
            rows, cols = pxy_normalized.shape
            for i in range(rows):
                for j in range(cols):
                    if (pxy_normalized[i, j] > 0 and px_normalized[i] > 0 and 
                        py_normalized[j] > 0):
                        mi += pxy_normalized[i, j] * np.log2(
                            pxy_normalized[i, j] / (px_normalized[i] * py_normalized[j]))
        except Exception as e:
            print(f"Warning: Error during MI calculation: {e}")
            print(f"Debug info - px_normalized shape: {px_normalized.shape}, py_normalized shape: {py_normalized.shape}")
            return 0.0
        
        return max(0, mi)  # Ensure non-negative MI due to numerical issues
    
    except Exception as e:
        print(f"Error in mutual_information: {e}")
        traceback.print_exc()
        return 0.0

def conditional_mutual_information(x, y, z, bins=10, preserve_structure=False):
    """
    Calculate conditional mutual information I(X;Y|Z) without using joint_entropy.
    
    Args:
        x, y, z: Input variables
        bins: Number of bins for discretization
        preserve_structure: If True, preserve multidimensional structure
        
    Returns:
        cmi: Conditional mutual information in bits
    """
    try:
        # Convert and validate inputs
        x = validate_and_convert(x, preserve_structure=False)
        y = validate_and_convert(y, preserve_structure=False)
        z = validate_and_convert(z, preserve_structure=False)
        
        # Make sure all arrays have the same number of samples
        min_length = min(len(x), len(y), len(z))
        x = x[:min_length]
        y = y[:min_length]
        z = z[:min_length]
        
        # For high-dimensional data, use only the first dimension or sample
        if x.ndim > 1 and x.shape[1] > 1:
            x = x[:, 0].reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0].reshape(-1, 1)
        if z.ndim > 1 and z.shape[1] > 1:
            # For conditioning variable, use a few dimensions
            z = z[:, :min(3, z.shape[1])]
        
        # Discretize the conditioning variable
        if z.ndim > 1 and z.shape[1] > 1:
            z_binned = np.zeros(z.shape[0], dtype=int)
            for i in range(z.shape[1]):
                z_dim = z[:, i]
                z_dim_binned = bin_data(z_dim, int(bins/z.shape[1]))
                z_binned += z_dim_binned * (bins**(i))
        else:
            z_binned = bin_data(z, bins)
        
        # Calculate CMI as weighted sum of MI for each Z value
        unique_z = np.unique(z_binned)
        cmi = 0.0
        total_samples = len(z_binned)
        
        for z_val in unique_z:
            # Get indices where Z has this value
            indices = (z_binned == z_val)
            # Calculate weight as proportion of samples with this Z value
            weight = np.sum(indices) / total_samples
            # Calculate MI conditioned on this Z value
            if np.sum(indices) > 1:  # Need at least 2 samples for MI calculation
                cmi += weight * mutual_information(x[indices], y[indices], bins)
        
        return max(0.0, cmi)
    
    except Exception as e:
        print(f"Error in conditional_mutual_information: {e}")
        traceback.print_exc()
        return 0.0

def pid_2x1y(x1data, x2data, ydata, bins=10):
    """
    Calculate Partial Information Decomposition (PID) for two sources and one target.
    
    This decomposes mutual information into unique, redundant, and synergistic components.
    
    Args:
        x1data: First source (X₁)
        x2data: Second source (X₂)
        ydata: Target (Y)
        bins: Number of bins for histogram
        
    Returns:
        Dictionary with PID components: unique1, unique2, redundancy, synergy
    """
    # Default return values in case of error
    default_result = {
        "unique1": 0.0,
        "unique2": 0.0,
        "redundancy": 0.0,
        "synergy": 0.0,
        "joint": 0.0
    }
    
    try:
        # Convert and validate inputs
        x1data = validate_and_convert(x1data)
        x2data = validate_and_convert(x2data)
        ydata = validate_and_convert(ydata)
        
        # Make sure all arrays have the same number of samples
        min_length = min(len(x1data), len(x2data), len(ydata))
        if len(x1data) != min_length or len(x2data) != min_length or len(ydata) != min_length:
            print(f"Warning: arrays have different lengths, truncating to {min_length}")
            x1data = x1data[:min_length]
            x2data = x2data[:min_length]
            ydata = ydata[:min_length]
        
        # Calculate individual mutual information
        i_x1y = mutual_information(x1data, ydata, bins=bins)
        i_x2y = mutual_information(x2data, ydata, bins=bins)
        
        if i_x1y is None or i_x2y is None:
            print("Warning: Mutual information calculation returned None")
            return default_result
        
        # Calculate joint mutual information
        i_x1x2y = mutual_information(np.column_stack((x1data, x2data)), ydata, bins=bins)
        
        if i_x1x2y is None:
            print("Warning: Joint mutual information calculation returned None")
            return default_result
        
        # Calculate interaction information (can be negative)
        ii = i_x1x2y - i_x1y - i_x2y
        
        # Estimate redundancy using minimum mutual information
        redundancy = min(i_x1y, i_x2y)
        
        # Calculate unique information
        unique1 = i_x1y - redundancy
        unique2 = i_x2y - redundancy
        
        # Calculate synergy
        synergy = i_x1x2y - i_x1y - i_x2y + redundancy
        
        return {
            "unique1": max(0, unique1),
            "unique2": max(0, unique2),
            "redundancy": max(0, redundancy),
            "synergy": max(0, synergy),
            "joint": i_x1x2y
        }
    
    except Exception as e:
        print(f"Error calculating PID: {e}")
        return default_result

def transfer_entropy(source, target, k=1, lag=1, bins=10):
    """
    Calculate transfer entropy from source to target.
    
    Transfer entropy measures the directed flow of information.
    
    Args:
        source: Source time series
        target: Target time series
        k: History length
        lag: Time lag
        bins: Number of bins for histogram
        
    Returns:
        te: Transfer entropy in bits
    """
    source = validate_and_convert(source)
    target = validate_and_convert(target)
    
    if len(source) != len(target):
        min_length = min(len(source), len(target))
        source = source[:min_length]
        target = target[:min_length]
    
    # Ensure sources and targets are 1D
    if source.ndim > 1:
        source = source.flatten()
    if target.ndim > 1:
        target = target.flatten()
    
    n = len(target) - lag - k
    
    if n <= 0:
        print(f"Error: Time series too short for lag={lag} and history={k}")
        return 0.0
    
    # Create lagged and history arrays
    target_future = target[lag+k:]
    target_history = np.zeros((n, k))
    for i in range(k):
        target_history[:, i] = target[k-i-1:n+k-i-1]
    
    source_present = source[k:n+k]
    
    # Calculate transfer entropy as conditional mutual information
    te = conditional_mutual_information(source_present, target_future, target_history, bins=bins)
    
    return te

def bin_data(x, bins=10):
    """
    Bin continuous data into discrete bins.
    
    Args:
        x: Input data (array-like)
        bins: Number of bins or bin edges
        
    Returns:
        Binned data as integers
    """
    # Convert input to numpy array if it's not already
    x = np.asarray(x)
    
    # Handle multidimensional arrays - not suitable for np.bincount
    if x.ndim > 1:
        print(f"Warning: bin_data received {x.ndim}D data with shape {x.shape}. Should use histogramdd for multidimensional data.")
        # For direct use with bincount later, we need to flatten this
        # We'll return a flattened version with a warning
        x = x.flatten()
    
    # Handle empty arrays
    if x.size == 0:
        return np.array([], dtype=int)
    
    # Handle the case when bins is None
    if bins is None:
        bins = 10  # Default to 10 bins
    
    if np.isscalar(bins):
        # Check if all values are the same or if there are NaN values
        if np.all(x == x[0]) or np.any(np.isnan(x)):
            return np.zeros_like(x, dtype=int)
        
        # Use percentile-based binning to handle outliers better
        try:
            bin_edges = np.percentile(x, np.linspace(0, 100, int(bins)+1))
            # Ensure unique edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) <= 1:  # Handle constant data
                return np.zeros_like(x, dtype=int)
        except Exception as e:
            print(f"Error in bin_data percentile calculation: {e}")
            # Fallback to evenly spaced bins
            x_min, x_max = np.min(x), np.max(x)
            if x_min == x_max:  # All values are the same
                return np.zeros_like(x, dtype=int)
            bin_edges = np.linspace(x_min, x_max, int(bins)+1)
    else:
        # If bins is already an array of bin edges
        bin_edges = np.asarray(bins)
        if bin_edges.size <= 1:
            return np.zeros_like(x, dtype=int)
    
    # Ensure bin_edges has at least 2 elements for digitize
    if len(bin_edges) < 2:
        # If there's only one edge or empty, create a simple binary binning
        x_min, x_max = np.min(x), np.max(x)
        if x_min == x_max:  # All values are the same
            return np.zeros_like(x, dtype=int)
        bin_edges = np.array([x_min, x_max])
    
    # Digitize data into bins
    try:
        binned = np.digitize(x, bin_edges[1:])
        return binned
    except Exception as e:
        print(f"Error in np.digitize: {e}, x.shape={x.shape}, bin_edges={bin_edges}")
        # Last resort fallback
        return np.zeros_like(x, dtype=int)

def entropy(x, bins=10):
    """
    Calculate Shannon entropy of a random variable with improved handling for high dimensions.
    
    Args:
        x: Input data (array-like)
        bins: Number of bins for discretization
        
    Returns:
        Entropy value in bits (base 2)
    """
    # Convert to numpy array
    x = np.asarray(x)
    
    # Different handling based on data dimensionality
    if x.ndim > 1:
        # For multi-dimensional data, check if dimensions are too high
        if x.shape[1] > 20:  # If more than 20 dimensions, use alternative approach
            print(f"High dimensional entropy calculation: {x.shape}, using dimension reduction")
            try:
                from sklearn.decomposition import PCA
                # Reduce to a more manageable number of dimensions
                n_components = min(10, x.shape[0]//5)
                pca = PCA(n_components=max(2, n_components))
                x_reduced = pca.fit_transform(x)
                print(f"Reduced from {x.shape} to {x_reduced.shape}")
                print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2f}")
                
                # Calculate entropy on reduced data
                hist, _ = np.histogramdd(x_reduced, bins=bins)
            except ImportError:
                print("PCA reduction failed, using dimension sampling")
                # Sample a subset of dimensions randomly
                import random
                random.seed(42)
                sampled_dims = random.sample(range(x.shape[1]), min(10, x.shape[1]))
                x_sampled = x[:, sampled_dims]
                
                # Calculate entropy on sampled dimensions
                hist, _ = np.histogramdd(x_sampled, bins=bins)
        else:
            # For data with moderate dimensions, use histogramdd directly
            # but with fewer bins for higher dimensions
            adjusted_bins = max(2, int(bins / (x.shape[1]**0.5)))  # Reduce bins for higher dimensions
            try:
                hist, _ = np.histogramdd(x, bins=adjusted_bins)
            except Exception as e:
                print(f"Error in np.histogramdd: {e}. Falling back to sampling dimensions.")
                # Sample a subset of dimensions randomly
                import random
                random.seed(42)
                sampled_dims = random.sample(range(x.shape[1]), min(5, x.shape[1]))
                x_sampled = x[:, sampled_dims]
                
                # Calculate entropy on sampled dimensions
                hist, _ = np.histogramdd(x_sampled, bins=bins)
        
        # Normalize and calculate entropy
        hist = hist.astype(float) / np.sum(hist)
        hist = hist[hist > 0]  # Remove zeros
        return -np.sum(hist * np.log2(hist))
    else:
        # For 1D data, use bincount approach as before
        try:
            x_binned = bin_data(x, bins)
            counts = np.bincount(x_binned)
            # Normalize to get probability distribution
            probs = counts[counts > 0].astype(float) / len(x_binned)
            # Compute entropy
            return -np.sum(probs * np.log2(probs))
        except Exception as e:
            print(f"Error in 1D entropy calculation: {e}")
            # Fallback to numpy histogram
            try:
                hist, _ = np.histogram(x, bins=bins)
                hist = hist.astype(float) / np.sum(hist)
                hist = hist[hist > 0]
                return -np.sum(hist * np.log2(hist))
            except Exception as e2:
                print(f"Error in entropy fallback: {e2}")
                return 0.0

def visualize_data_shape(name, data):
    """
    Debug utility to visualize data shape and type information.
    
    Args:
        name: String identifier for the data
        data: The data to visualize
    """
    if isinstance(data, np.ndarray):
        print(f"Data '{name}': shape={data.shape}, dtype={data.dtype}, min={np.min(data) if data.size > 0 else 'N/A'}, max={np.max(data) if data.size > 0 else 'N/A'}")
    elif isinstance(data, list):
        print(f"Data '{name}': list of len={len(data)}, type={type(data[0]) if data else 'empty'}")
    else:
        print(f"Data '{name}': type={type(data)}")

def joint_entropy(x, y, bins=10):
    """
    Calculate joint entropy of two random variables with improved handling for high-dimensional data.
    
    Args:
        x, y: Input data (array-like)
        bins: Number of bins for discretization
        
    Returns:
        Joint entropy value in bits
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Debug shapes
    visualize_data_shape("x in joint_entropy", x)
    visualize_data_shape("y in joint_entropy", y)
    
    # Check if dimensions are too high for histogramdd
    high_dims = False
    if x.ndim > 1 and y.ndim > 1:
        total_dims = x.shape[1] + y.shape[1]
        if total_dims > 20:  # If total dimensions exceed 20, use alternative approach
            high_dims = True
            print(f"High dimensional joint entropy calculation: total dims={total_dims}, using PCA reduction")
    
    # Check if x or y is multidimensional
    if (x.ndim > 1 and x.shape[1] > 1) or (y.ndim > 1 and y.shape[1] > 1):
        print(f"Joint entropy with multidimensional data: x.shape={x.shape}, y.shape={y.shape}")
        
        # Ensure both are 2D arrays for proper stacking
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Make sure both have same number of samples
        min_samples = min(x.shape[0], y.shape[0])
        if x.shape[0] != y.shape[0]:
            print(f"Warning: Different number of samples in x ({x.shape[0]}) and y ({y.shape[0]}), truncating to {min_samples}")
            x = x[:min_samples]
            y = y[:min_samples]
        
        # Stack data and calculate joint histogram
        try:
            joint_data = np.hstack((x, y))
            visualize_data_shape("joint_data", joint_data)
            
            # Use dimension reduction if data is too high-dimensional
            if high_dims:
                try:
                    from sklearn.decomposition import PCA
                    # Reduce to a more manageable number of dimensions
                    pca = PCA(n_components=min(10, joint_data.shape[0]//5))
                    joint_data_reduced = pca.fit_transform(joint_data)
                    print(f"Reduced joint data from {joint_data.shape} to {joint_data_reduced.shape}")
                    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2f}")
                    
                    # Use the reduced data for histogram calculation
                    hist, _ = np.histogramdd(joint_data_reduced, bins=bins)
                except ImportError:
                    print("PCA reduction failed, using binning strategy")
                    # Alternative: Use pairwise entropy and make an approximation
                    joint_entropy_val = 0
                    for i in range(min(10, joint_data.shape[1])):
                        joint_entropy_val += entropy(joint_data[:, i], bins)
                    joint_entropy_val /= min(10, joint_data.shape[1])
                    return joint_entropy_val
            else:
                # For lower dimensions, use histogramdd directly
                hist, _ = np.histogramdd(joint_data, bins=min(bins, 5))  # Use fewer bins for higher dims
            
            # Normalize to get probability distribution
            hist = hist.astype(float) / np.sum(hist)
            
            # Remove zeros
            hist = hist[hist > 0]
            
            # Calculate entropy
            return -np.sum(hist * np.log2(hist))
        except Exception as e:
            print(f"Error in joint_entropy histogramdd: {e}")
            # Fallback to calculating entropy separately
            try:
                h_x = entropy(x, bins)
                h_y = entropy(y, bins)
                # Estimate joint entropy with a correction factor
                # In independent case, H(X,Y) = H(X) + H(Y)
                # Adding a small discount as a heuristic
                return h_x + h_y - 0.1*min(h_x, h_y)  # Apply a small correction
            except Exception as e2:
                print(f"Error in joint_entropy fallback: {e2}")
                return 0.0
    
    # For 1D data, use the original method with binning
    try:
        x_binned = bin_data(x, bins)
        y_binned = bin_data(y, bins)
        
        # Calculate max bin indices
        x_max = max(np.max(x_binned) + 1, bins)
        y_max = max(np.max(y_binned) + 1, bins)
        
        # Create joint histogram
        joint_counts = np.zeros((x_max, y_max))
        
        # Fill the joint histogram
        for i in range(len(x_binned)):
            joint_counts[x_binned[i], y_binned[i]] += 1
        
        # Calculate joint probabilities
        joint_probs = joint_counts / len(x_binned)
        joint_probs = joint_probs[joint_probs > 0]
        
        return -np.sum(joint_probs * np.log2(joint_probs))
    except Exception as e:
        print(f"Error in joint_entropy: {e}")
        
        # Fallback to hist_2d
        try:
            hist, _, _ = hist_2d(x, y, bins=bins)
            hist = hist.astype(float) / np.sum(hist)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except Exception as e2:
            print(f"Error in joint_entropy fallback: {e2}")
            return 0.0

def mutual_information(x, y, bins=10):
    """
    Calculate mutual information between two random variables.
    
    Args:
        x, y: Input data (array-like)
        bins: Number of bins for discretization
        
    Returns:
        Mutual information in bits
    """
    # Instead of using joint_entropy, calculate MI directly
    # Convert inputs to numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Ensure proper dimensions
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    # Make sure both have same number of samples
    min_samples = min(x.shape[0], y.shape[0])
    x = x[:min_samples]
    y = y[:min_samples]
    
    # Use np.histogram2d directly for joint distribution
    try:
        # For higher dimensions, use first columns
        x_col = x[:, 0] if x.shape[1] > 0 else x.flatten()
        y_col = y[:, 0] if y.shape[1] > 0 else y.flatten()
        
        # Calculate joint histogram
        joint_hist, x_edges, y_edges = np.histogram2d(x_col, y_col, bins=bins)
        
        # Calculate marginal histograms
        x_hist = np.sum(joint_hist, axis=1)
        y_hist = np.sum(joint_hist, axis=0)
        
        # Normalize to get probability distributions
        joint_prob = joint_hist / np.sum(joint_hist)
        x_prob = x_hist / np.sum(x_hist)
        y_prob = y_hist / np.sum(y_hist)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(len(x_prob)):
            for j in range(len(y_prob)):
                if joint_prob[i, j] > 0 and x_prob[i] > 0 and y_prob[j] > 0:
                    mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (x_prob[i] * y_prob[j]))
        
        return max(0, mi)  # Ensure non-negative due to numerical issues
    
    except Exception as e:
        print(f"Error in mutual_information: {e}")
        traceback.print_exc()
        return 0.0

def conditional_entropy(x, y, bins=10):
    """
    Calculate conditional entropy H(X|Y).
    
    Args:
        x: Target variable
        y: Conditioning variable
        bins: Number of bins for discretization
        
    Returns:
        Conditional entropy in bits
    """
    # H(X|Y) = H(X,Y) - H(Y)
    return joint_entropy(x, y, bins) - entropy(y, bins)

def conditional_mutual_information(x, y, z, bins=10):
    """
    Calculate conditional mutual information I(X;Y|Z) without using joint_entropy.
    
    Args:
        x, y, z: Input variables
        bins: Number of bins for discretization
        
    Returns:
        Conditional mutual information in bits
    """
    try:
        # Convert and validate inputs
        x = validate_and_convert(x, preserve_structure=False)
        y = validate_and_convert(y, preserve_structure=False)
        z = validate_and_convert(z, preserve_structure=False)
        
        # Make sure all arrays have the same number of samples
        min_length = min(len(x), len(y), len(z))
        x = x[:min_length]
        y = y[:min_length]
        z = z[:min_length]
        
        # For high-dimensional data, use only the first dimension or sample
        if x.ndim > 1 and x.shape[1] > 1:
            x = x[:, 0].reshape(-1, 1)
        if y.ndim > 1 and y.shape[1] > 1:
            y = y[:, 0].reshape(-1, 1)
        if z.ndim > 1 and z.shape[1] > 1:
            # For conditioning variable, use a few dimensions
            z = z[:, :min(3, z.shape[1])]
        
        # Discretize the conditioning variable
        if z.ndim > 1 and z.shape[1] > 1:
            z_binned = np.zeros(z.shape[0], dtype=int)
            for i in range(z.shape[1]):
                z_dim = z[:, i]
                z_dim_binned = bin_data(z_dim, int(bins/z.shape[1]))
                z_binned += z_dim_binned * (bins**(i))
        else:
            z_binned = bin_data(z, bins)
        
        # Calculate CMI as weighted sum of MI for each Z value
        unique_z = np.unique(z_binned)
        cmi = 0.0
        total_samples = len(z_binned)
        
        for z_val in unique_z:
            # Get indices where Z has this value
            indices = (z_binned == z_val)
            # Calculate weight as proportion of samples with this Z value
            weight = np.sum(indices) / total_samples
            # Calculate MI conditioned on this Z value
            if np.sum(indices) > 1:  # Need at least 2 samples for MI calculation
                cmi += weight * mutual_information(x[indices], y[indices], bins)
        
        return max(0.0, cmi)
    
    except Exception as e:
        print(f"Error in conditional_mutual_information: {e}")
        traceback.print_exc()
        return 0.0

def interaction_information(x, y, z, bins=10):
    """
    Calculate interaction information I(X;Y;Z).
    
    Args:
        x, y, z: Input variables
        bins: Number of bins for discretization
        
    Returns:
        Interaction information in bits
    """
    # I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
    mi_xy = mutual_information(x, y, bins)
    cmi_xy_z = conditional_mutual_information(x, y, z, bins)
    return mi_xy - cmi_xy_z

def pid_2x1y(x1, x2, y, bins=10):
    """
    Calculate Partial Information Decomposition (PID) for two source variables
    and one target variable.
    
    Implements Williams & Beer's method to calculate:
    - unique information from each source
    - redundant information (shared between sources)
    - synergistic information (only available when considering both sources together)
    
    Args:
        x1, x2: Source variables
        y: Target variable
        bins: Number of bins for discretization
        
    Returns:
        Dictionary with PID components: 'unique1', 'unique2', 'redundancy', 'synergy'
    """
    # Calculate basic information quantities
    i_x1y = mutual_information(x1, y, bins)
    i_x2y = mutual_information(x2, y, bins)
    
    # Calculate joint mutual information
    x_joint = np.column_stack((x1, x2))
    i_xy = mutual_information_joint(x_joint, y, bins)
    
    # Estimate redundancy using minimum information
    if isinstance(y, np.ndarray) and y.ndim > 1:
        # Handle multivariate y by calculating MI for each variable and summing
        i_x1y_parts = [mutual_information(x1, y_i, bins) for y_i in y.T]
        i_x2y_parts = [mutual_information(x2, y_i, bins) for y_i in y.T]
        redundancy = sum(min(i1, i2) for i1, i2 in zip(i_x1y_parts, i_x2y_parts))
    else:
        # Minimum mutual information (MMI) method for redundancy
        redundancy = min(i_x1y, i_x2y)
    
    # Calculate unique information for each source
    unique1 = i_x1y - redundancy
    unique2 = i_x2y - redundancy
    
    # Calculate synergy
    synergy = i_xy - unique1 - unique2 - redundancy
    
    # Handle small numerical errors
    unique1 = max(0, unique1)
    unique2 = max(0, unique2)
    redundancy = max(0, redundancy)
    synergy = max(0, synergy)
    
    return {
        "unique1": unique1,
        "unique2": unique2,
        "redundancy": redundancy,
        "synergy": synergy
    }

def mutual_information_joint(xs, y, bins=10):
    """
    Calculate mutual information between joint variable X (multivariate) and Y.
    
    Args:
        xs: Joint variable as 2D array (samples, variables)
        y: Target variable
        bins: Number of bins for discretization
        
    Returns:
        Mutual information in bits
    """
    # Handle None or empty inputs
    if xs is None or y is None or np.size(xs) == 0 or np.size(y) == 0:
        print("Warning: Empty input data in mutual_information_joint")
        return 0.0
        
    # Convert inputs to numpy arrays
    xs = np.asarray(xs)
    y = np.asarray(y)
    
    # Check if lengths match
    if len(xs) != len(y):
        min_len = min(len(xs), len(y))
        print(f"Warning: Input length mismatch in mutual_information_joint. Truncating to {min_len}")
        xs = xs[:min_len]
        y = y[:min_len]
    
    # Special handling for multidimensional Y
    if y.ndim > 1 and y.shape[1] > 1:
        print(f"Processing multidimensional Y with shape {y.shape} in mutual_information_joint")
        try:
            # Calculate joint entropy and marginal entropies directly
            # H(X,Y) - H(X) - H(Y)
            joint_data = np.hstack((xs, y))
            h_xy = entropy(joint_data, bins=bins)
            h_x = entropy(xs, bins=bins)
            h_y = entropy(y, bins=bins)
            
            # Mutual information
            mi = h_x + h_y - h_xy
            return max(0, mi)  # Ensure non-negative
        except Exception as e:
            print(f"Error processing multidimensional Y: {e}")
            traceback.print_exc()
            return 0.0
    
    # Handle multivariate X
    if xs.ndim > 1 and xs.shape[1] > 1:
        try:
            # Flatten the joint state space
            x_dims = xs.shape[1]
            x_binned = np.zeros(len(xs), dtype=int)
            
            # Bin each dimension separately
            bin_multiplier = 1
            for dim in range(x_dims):
                x_dim = xs[:, dim]
                x_dim_binned = bin_data(x_dim, bins)
                # Combine bins using different powers
                x_binned += x_dim_binned * bin_multiplier
                bin_multiplier *= max(np.max(x_dim_binned) + 1, bins)
            
            # Now calculate mutual information with the combined variable
            if y.ndim > 1 and y.shape[1] > 1:
                # Handle multivariate Y
                y_dims = y.shape[1]
                y_binned = np.zeros(len(y), dtype=int)
                
                bin_multiplier = 1
                for dim in range(y_dims):
                    y_dim = y[:, dim]
                    y_dim_binned = bin_data(y_dim, bins)
                    y_binned += y_dim_binned * bin_multiplier
                    bin_multiplier *= max(np.max(y_dim_binned) + 1, bins)
                    
                return mutual_information(x_binned, y_binned, bins=max(bins, 1))
            else:
                # Y is univariate
                y_binned = bin_data(y, bins)
                return mutual_information(x_binned, y_binned, bins=max(bins, 1))
        except Exception as e:
            print(f"Error in mutual_information_joint: {e}")
            traceback.print_exc()
            return 0.0
    else:
        # X is univariate
        try:
            # Ensure xs is 1D if it's a single column
            if xs.ndim > 1:
                xs = xs.flatten()
            return mutual_information(xs, y, bins)
        except Exception as e:
            print(f"Error in mutual_information_joint (univariate case): {e}")
            traceback.print_exc()
            return 0.0

# Add aliases for compatibility with causal_emergence.py 
def mutual_info(x, y, bins=10):
    """Alias for mutual_information function to maintain compatibility with causal_emergence.py"""
    print(f"mutual_info called: x shape: {x.shape}, y shape: {y.shape}")
    result = mutual_information(x, y, bins=bins)
    print(f"mutual_info result: {result}")
    return result

def conditional_mutual_info(x, y, z, bins=10):
    """Alias for conditional_mutual_information function to maintain compatibility with causal_emergence.py"""
    print(f"conditional_mutual_info called: x shape: {x.shape}, y shape: {y.shape}, z shape: {z.shape}")
    result = conditional_mutual_information(x, y, z, bins=bins)
    print(f"conditional_mutual_info result: {result}")
    return result

import os
import subprocess
import json
from typing import Dict, Optional, List, Tuple

def get_gpu_memory_info() -> List[Dict[str, int]]:
    """
    Get memory information for all available GPUs.
    
    Returns:
        List of dicts with memory info for each GPU (total, used, free memory in MiB)
    """
    try:
        # Try nvidia-smi for NVIDIA GPUs
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'], 
            encoding='utf-8'
        )
        
        # Parse the output
        gpu_info = []
        for i, line in enumerate(result.strip().split('\n')):
            total, used, free = map(int, line.split(','))
            gpu_info.append({
                'index': i, 
                'total': total, 
                'used': used, 
                'free': free, 
                'utilization': round(used / total * 100, 1)
            })
        return gpu_info
        
    except (subprocess.SubprocessError, FileNotFoundError):
        # NVIDIA tools not available, try AMD ROCm tools
        try:
            result = subprocess.check_output(['rocm-smi', '--showmeminfo', 'vram', '--json'], encoding='utf-8')
            data = json.loads(result)
            
            gpu_info = []
            for i, card in enumerate(data.get('cards', [])):
                memory = card.get('memory usage', {})
                total = memory.get('total memory', 0)
                used = memory.get('used memory', 0)
                free = memory.get('free memory', 0)
                
                # Convert to MiB if values are in bytes
                if total > 10000000:  # Probably in bytes
                    total = total // 1024 // 1024
                    used = used // 1024 // 1024
                    free = free // 1024 // 1024
                
                gpu_info.append({
                    'index': i, 
                    'total': total, 
                    'used': used, 
                    'free': free, 
                    'utilization': round(used / total * 100, 1) if total > 0 else 0
                })
            return gpu_info
            
        except (subprocess.SubprocessError, FileNotFoundError, json.JSONDecodeError):
            # No GPU info available
            return []

def configure_gpu_memory_for_asal(
    device: Optional[str] = None,
    memory_fraction: float = 0.9,
    memory_growth: bool = True
) -> bool:
    """
    Configure GPU memory usage for ASAL simulations.
    
    Args:
        device: Specific device to use (e.g., 'cuda:0', 'cpu')
        memory_fraction: Fraction of GPU memory to allocate (0.0 to 1.0)
        memory_growth: Whether to use memory growth (dynamic allocation)
        
    Returns:
        True if configuration was successful, False otherwise
    """
    # Set JAX default platform/device if specified
    if device is not None:
        if device.startswith('cuda'):
            os.environ['JAX_PLATFORM_NAME'] = 'gpu'
            if ':' in device:
                os.environ['CUDA_VISIBLE_DEVICES'] = device.split(':')[1]
        elif device == 'cpu':
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    
    # Configure TensorFlow (for CLIP model if it uses TF)
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if memory_fraction < 1.0:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(memory_fraction * 1024))]  # Memory in MB
                    )
            print(f"TensorFlow GPU memory configured: growth={memory_growth}, fraction={memory_fraction}")
            return True
    except ImportError:
        pass
    except RuntimeError as e:
        print(f"TensorFlow GPU configuration failed: {e}")
    
    # Configure JAX memory
    try:
        import jax
        if memory_fraction < 1.0:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(memory_fraction)
        print(f"JAX memory fraction set to {memory_fraction}")
        return True
    except ImportError:
        pass
    
    return False

def estimate_batch_size_for_asal(
    simulation_type: str, 
    grid_size: int = 32,
    base_memory_usage: int = 2048,  # Base memory usage in MiB
    safety_factor: float = 0.8  # Safety factor to avoid OOM
) -> int:
    """
    Estimate appropriate batch size based on available GPU memory.
    
    Args:
        simulation_type: Type of simulation ('gol', 'boids', 'lenia')
        grid_size: Size of simulation grid
        base_memory_usage: Base memory usage in MiB
        safety_factor: Safety factor to apply to available memory
        
    Returns:
        Recommended batch size
    """
    # Get available GPU memory
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        # No GPU info available, return a conservative default
        return 8
    
    # Find GPU with most free memory
    best_gpu = max(gpu_info, key=lambda x: x['free'])
    free_memory = best_gpu['free'] * safety_factor  # Apply safety factor
    
    # Estimate memory per simulation based on simulation_type and grid_size
    if simulation_type == 'gol':
        memory_per_sim = base_memory_usage + (grid_size ** 2 * 0.1)
    elif simulation_type == 'boids':
        memory_per_sim = base_memory_usage + (grid_size ** 2 * 0.05)
    elif simulation_type == 'lenia':
        memory_per_sim = base_memory_usage + (grid_size ** 2 * 0.2)
    else:
        memory_per_sim = base_memory_usage + (grid_size ** 2 * 0.15)  # Default estimate
    
    # Calculate and return recommended batch size
    batch_size = max(1, int(free_memory / memory_per_sim))
    return batch_size

def print_gpu_usage_summary() -> None:
    """Print a summary of current GPU memory usage."""
    gpu_info = get_gpu_memory_info()
    
    if not gpu_info:
        print("No GPU information available")
        return
        
    print("\nGPU Memory Usage:")
    print("------------------")
    for gpu in gpu_info:
        print(f"GPU {gpu['index']}: {gpu['used']}/{gpu['total']} MiB ({gpu['utilization']}%)")
    print("------------------")

# 結果保存用のユーティリティ関数を追加
def create_results_archive(results_dir: str, archive_name: Optional[str] = None) -> str:
    """
    Create a compressed archive of results for backup or sharing.
    
    Args:
        results_dir: Directory containing results to archive
        archive_name: Optional name for the archive (default: auto-generate)
        
    Returns:
        Path to the created archive file
    """
    import shutil
    import tempfile
    
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Generate archive name if not provided
    if archive_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(results_dir)
        archive_name = f"{base_name}_{timestamp}"
    
    # Create the archive
    archive_path = os.path.join(tempfile.gettempdir(), f"{archive_name}.zip")
    
    try:
        shutil.make_archive(
            os.path.splitext(archive_path)[0],  # Base name (without extension)
            'zip',                              # Format
            results_dir                         # Root directory to archive
        )
        
        # Move to the results directory
        final_path = os.path.join(os.path.dirname(results_dir), f"{archive_name}.zip")
        shutil.move(archive_path, final_path)
        return final_path
    except Exception as e:
        print(f"Error creating archive: {e}")
        return ""

def save_gpu_info_to_file(output_path: str) -> None:
    """
    Save current GPU information to a file.
    
    Args:
        output_path: Path to save the GPU info
    """
    gpu_info = get_gpu_memory_info()
    
    with open(output_path, 'w') as f:
        f.write("GPU Information\n")
        f.write("==============\n\n")
        
        if not gpu_info:
            f.write("No GPU information available\n")
            return
        
        for gpu in gpu_info:
            f.write(f"GPU {gpu['index']}:\n")
            f.write(f"  Total Memory: {gpu['total']} MiB\n")
            f.write(f"  Used Memory:  {gpu['used']} MiB\n")
            f.write(f"  Free Memory:  {gpu['free']} MiB\n")
            f.write(f"  Utilization:  {gpu['utilization']}%\n\n")

if __name__ == "__main__":
    # Example usage
    print_gpu_usage_summary()
    batch_size = estimate_batch_size_for_asal('gol', grid_size=64)
    print(f"Recommended batch size for GoL (64x64): {batch_size}")

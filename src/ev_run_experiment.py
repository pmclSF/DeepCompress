import os
import yaml
from subprocess import Popen, PIPE
from typing import Dict, List, Any
import logging

def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required paths
    required_dirs = ['MPEG_TMC13_DIR', 'PCERROR', 'MPEG_DATASET_DIR', 'EXPERIMENT_DIR']
    for dir_key in required_dirs:
        if dir_key not in config:
            raise ValueError(f"Missing required configuration key: {dir_key}")
    
    return config

def prepare_experiment_params(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare parameters for each experiment configuration.
    
    Args:
        config (Dict[str, Any]): Loaded configuration dictionary
        
    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries for each experiment
    """
    experiment_params = []
    
    for model_config in config['model_configs']:
        model_id = model_config['id']
        
        for lambda_val in model_config['lambdas']:
            for data_config in config['data']:
                params = {
                    'model_id': model_id,
                    'lambda_val': lambda_val,
                    'pc_name': data_config['pc_name'],
                    'cfg_name': data_config['cfg_name'],
                    'input_pc': data_config['input_pc'],
                    'input_norm': data_config['input_norm'],
                    'output_dir': os.path.join(
                        config['EXPERIMENT_DIR'],
                        data_config['pc_name'],
                        model_id,
                        f"{lambda_val:.2e}"
                    )
                }
                
                # Add all configuration paths
                params.update({
                    'mpeg_dir': config['MPEG_TMC13_DIR'],
                    'pcerror': config['PCERROR'],
                    'dataset_dir': config['MPEG_DATASET_DIR']
                })
                
                experiment_params.append(params)
    
    return experiment_params

def run_experiment(params: Dict[str, Any], no_stream_redirection: bool = False):
    """
    Run a single experiment with the given parameters.
    
    Args:
        params (Dict[str, Any]): Parameters for the experiment
        no_stream_redirection (bool): If True, don't redirect stdout/stderr
    """
    os.makedirs(params['output_dir'], exist_ok=True)
    
    # Construct command for experiment
    cmd = [
        os.path.join(params['mpeg_dir'], 'build', 'tmc3'),
        '--config={}'.format(params['cfg_name']),
        '--uncompressedDataPath={}'.format(
            os.path.join(params['dataset_dir'], params['input_pc'])
        ),
        '--reconstructedDataPath={}'.format(
            os.path.join(params['output_dir'], 'rec.ply')
        ),
        '--modelPath={}'.format(params['model_id']),
        '--lambda={}'.format(params['lambda_val'])
    ]
    
    # Run the experiment
    if no_stream_redirection:
        process = Popen(cmd)
    else:
        log_file = os.path.join(params['output_dir'], 'log.txt')
        with open(log_file, 'w') as f:
            process = Popen(cmd, stdout=f, stderr=f)
    
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"Experiment failed with return code {process.returncode}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run compression experiments")
    parser.add_argument("config", help="Path to experiment configuration YAML")
    parser.add_argument("--no-redirect", action="store_true", 
                       help="Don't redirect stdout/stderr to log files")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_experiment_config(args.config)
    
    # Prepare experiment parameters
    experiment_params = prepare_experiment_params(config)
    
    # Run experiments
    for params in experiment_params:
        run_experiment(params, no_stream_redirection=args.no_redirect)
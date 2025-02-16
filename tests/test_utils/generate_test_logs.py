import os
from tests.test_utils.config_generator import generate_base_config, generate_edge_case_configs
from tests.test_utils.model_runner import ModelRunner
from tests.test_utils.log_capture import LogStore

def main():
    # Create output directories for logs
    os.makedirs("tests/test_data/logs/base_run", exist_ok=True)
    os.makedirs("tests/test_data/logs/edge_cases", exist_ok=True)
    
    # Run base configuration
    base_config = generate_base_config()
    runner = ModelRunner(base_config)
    result, metadata = runner.run_model()
    logs = runner.get_captured_logs()
    
    # Save logs for base run
    base_log_path = "tests/test_data/logs/base_run/base_run.log"
    store = LogStore(base_log_path)
    store.save_logs(logs)
    
    # Run edge case configurations
    edge_configs = generate_edge_case_configs()
    for idx, config in enumerate(edge_configs):
        runner = ModelRunner(config)
        result, metadata = runner.run_model()
        logs = runner.get_captured_logs()
        edge_log_path = f"tests/test_data/logs/edge_cases/edge_run_{idx}.log"
        store = LogStore(edge_log_path)
        store.save_logs(logs)
    
    print("Test logs generated successfully.")

if __name__ == "__main__":
    main()

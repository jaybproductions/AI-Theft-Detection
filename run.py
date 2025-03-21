import argparse
from utils.config import load_config
from interface.pipeline import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run real-time theft detection")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration settings
    config = load_config(args.config)

    # Run the detection pipeline
    run_inference(config)

if __name__ == '__main__':
    main()

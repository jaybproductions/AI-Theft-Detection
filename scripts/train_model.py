import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.action_recognition.train import train
from utils.config import load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    train(config)
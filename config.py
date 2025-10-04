"""
Config file for the voice emotion app
FallenGodfather
"""

import os
import json

class Config:
    # basic app info
    APP_NAME = "Voice Emotion AI"
    APP_VERSION = "2.0"
    AUTHOR = "FallenGodfather"

    # audio settings
    SAMPLE_RATE = 22050
    CHUNK_SIZE = 2048
    CHANNELS = 1
    RECORD_TIME = 20  # seconds

    # model settings
    EMOTIONS = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'disgusted']
    CNN_WEIGHT = 0.7
    KNN_WEIGHT = 0.3

    # feature params (optimized)
    N_MFCC = 12
    N_CHROMA = 8
    N_MEL = 64

    # colors for UI
    COLORS = {
        'primary': '#2c3e50',
        'secondary': '#34495e',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'spotify': '#1db954'
    }

    # folders
    MODELS_DIR = 'models'
    DATA_DIR = 'data'
    ASSETS_DIR = 'assets'

    @classmethod
    def setup_dirs(cls):
        """Create needed directories"""
        dirs = [cls.MODELS_DIR, cls.DATA_DIR, cls.ASSETS_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def save_config(cls, filename='config.json'):
        """Save config to file"""
        data = {
            'app': cls.APP_NAME,
            'version': cls.APP_VERSION,
            'sample_rate': cls.SAMPLE_RATE,
            'emotions': cls.EMOTIONS,
            'colors': cls.COLORS
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_config(cls, filename='config.json'):
        """Load config from file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return {}

if __name__ == "__main__":
    Config.setup_dirs()
    Config.save_config()
    print("Config ready")

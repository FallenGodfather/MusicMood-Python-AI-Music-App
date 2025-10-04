#!/usr/bin/env python3
"""
Setup script for Voice Emotion AI
FallenGodfather
"""

import os
import sys
import subprocess
import platform

def run_cmd(cmd):
    """Run a command"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_system_stuff():
    """Install system dependencies"""
    print("Installing system dependencies...")

    system = platform.system().lower()

    if system == "linux":
        cmds = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-dev python3-pip",
            "sudo apt-get install -y portaudio19-dev libasound2-dev"
        ]
    elif system == "darwin":  # macOS
        cmds = ["brew install portaudio"]
    else:
        print("Windows: Please install Python 3.8+ manually")
        return True

    for cmd in cmds:
        print(f"Running: {cmd}")
        success, _, err = run_cmd(cmd)
        if not success:
            print(f"Warning: {cmd} failed - {err}")

    return True

def install_python_deps():
    """Install Python packages"""
    print("Installing Python packages...")

    success, _, err = run_cmd("pip install -r requirements.txt")
    if not success:
        print(f"Failed to install packages: {err}")
        return False

    print("Packages installed!")
    return True

def setup_dirs():
    """Create directories"""
    print("Setting up directories...")

    dirs = ["models", "data", "assets"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print("Directories ready!")
    return True

def test_imports():
    """Test if everything imports"""
    print("Testing imports...")

    imports = [
        "kivy", "numpy", "tensorflow", 
        "librosa", "sklearn", "spotipy"
    ]

    for pkg in imports:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} failed")
            return False

    print("All imports work!")
    return True

def main():
    """Main setup"""
    print("Voice Emotion AI - Setup")
    print("=" * 30)

    steps = [
        ("System dependencies", install_system_stuff),
        ("Python packages", install_python_deps),
        ("Directories", setup_dirs),
        ("Testing imports", test_imports)
    ]

    for name, func in steps:
        print(f"\n{name}...")
        if not func():
            print(f"Setup failed at: {name}")
            sys.exit(1)

    print("\n" + "=" * 30)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Configure Spotify API (optional)")
    print("2. Run: python run_app.py")
    print("3. For Android: buildozer android debug")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple launcher for Voice Emotion AI
FallenGodfather
"""

import sys

def check_deps():
    """Quick dependency check"""
    try:
        import kivy
        import numpy
        print("Dependencies look good")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Run: python setup.py")
        return False

def main():
    print("Voice Emotion AI")
    print("FallenGodfather")
    print("-" * 20)

    if not check_deps():
        sys.exit(1)

    print("Starting app...")

    try:
        from main import VoiceEmotionApp
        app = VoiceEmotionApp()
        app.run()
    except KeyboardInterrupt:
        print("\nBye!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

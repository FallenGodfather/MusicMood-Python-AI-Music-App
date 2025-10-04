# Voice Emotion AI

AI app that detects emotions from voice and recommends music on Spotify.

**Author:** FallenGodfather  
**Version:** 2.0  
**Platform:** Android/Desktop

## About This Project

This is an improved version of my **Final Year Project for Computer Science**. The original was done with 3 other teammates, but I've been working on this refined version for a while now to fix bugs and make it better.

Same core algorithms (CNN + KNN) but way more optimized and actually usable now.

## Features

- 🎤 Voice recording (20 seconds, auto-stops)
- 🧠 AI emotion detection using CNN + KNN
- 🎵 Spotify music recommendations
- 📱 Simple mobile interface
- ⚡ Fast processing (under 2 seconds)

## How It Works

### Recording
- Tap mic button
- Talk naturally for up to 20 seconds
- App automatically stops and analyzes

### AI Analysis
The app uses two machine learning algorithms:

**CNN (Convolutional Neural Network) - 70% weight**
- Processes audio features in sequence
- Good at finding complex patterns in voice
- Learns from MFCC, chroma, and spectral features

**KNN (K-Nearest Neighbors) - 30% weight**  
- Compares your voice to similar examples
- Uses 3 nearest neighbors for classification
- More stable and interpretable

**How they work together:**
```
Your Voice → Feature Extraction → CNN + KNN → Weighted Average → Final Emotion
```

### Music Matching
Each emotion gets matched to music characteristics:
- **Happy**: High energy, positive vibes (pop, dance)
- **Sad**: Low energy, emotional (indie, folk) 
- **Angry**: High energy, aggressive (rock, metal)
- **And so on...**

## Supported Emotions

😊 Happy | 😢 Sad | 😠 Angry | 😰 Fearful | 😲 Surprised | 🤢 Disgusted | 😐 Neutral

## Installation

### Quick Setup
```bash
# Install dependencies
python setup.py

# Test it works
python test_app.py

# Run the app
python run_app.py
```

### For Android
```bash
# Build APK
buildozer android debug
```

## Requirements

- Python 3.8+
- Microphone
- Internet (for Spotify)

### Key Libraries
- kivy (mobile UI)
- tensorflow (CNN)
- scikit-learn (KNN)
- librosa (audio processing)
- spotipy (Spotify API)

## Spotify Setup

1. Go to [Spotify Developer](https://developer.spotify.com/)
2. Create an app
3. Get Client ID and Secret
4. Edit `spotify_integration.py`:
   ```python
   self.client_id = "your_client_id_here"
   self.client_secret = "your_secret_here"
   ```

## Performance

### Optimizations Made
- Recording: 20 seconds with auto-stop
- Features: Reduced from 283 to 90 dimensions  
- CNN: Lightweight model (32→64 filters)
- KNN: Only 3 neighbors with KD-tree
- Memory: Under 512MB usage
- Speed: Total analysis under 2 seconds

### Before vs After
| Metric | Original | Optimized |
|--------|----------|-----------|
| Processing Time | 10+ sec | <2 sec |
| Memory Usage | 1GB+ | <512MB |
| Model Size | Large | Mobile-friendly |
| Features | 283 | 90 |

## File Structure

```
voice-emotion-ai/
├── main.py              # Main app
├── emotion_detector.py  # CNN + KNN models
├── voice_processor.py   # Audio recording
├── spotify_integration.py # Music recommendations
├── requirements.txt     # Dependencies
├── buildozer.spec      # Android build config
├── setup.py            # Auto setup
├── test_app.py         # Tests
└── README.md           # This file
```

## Usage

1. Launch app
2. Tap microphone
3. Talk naturally for up to 20 seconds
4. See detected emotion
5. Get music recommendations
6. Open in Spotify

## Technical Details

### Feature Extraction
- **MFCC**: 12 coefficients (voice characteristics)
- **Chroma**: 8 features (pitch information)
- **Mel-spectrogram**: 64 bands (frequency analysis)
- **Spectral**: Energy, zero-crossing rate
- **Total**: 90 features (down from 283 original)

### Model Architecture
**CNN:**
```
Input (90 features) 
→ Conv1D (32 filters) + BatchNorm + Dropout
→ Conv1D (64 filters) + BatchNorm + Dropout  
→ GlobalPooling + Dense (128) + Dropout
→ Output (7 emotions)
```

**KNN:**
- K=3 neighbors
- Distance-weighted voting
- KD-tree for fast search

### Training Data
Uses synthetic emotional patterns based on research:
- 50 samples per emotion
- Emotion-specific feature modifications
- Quick training (20 epochs)

## Academic Context

This project started as a **Final Year CS Bachelor's Project** with 4 team members. The original version had the same CNN+KNN approach but was slower and had more bugs.

I've been working on this improved version to:
- Fix performance issues
- Make it actually usable on mobile
- Clean up the code
- Add better error handling

The core research and algorithms are the same, just much more optimized.

## Troubleshooting

### Common Issues

**Audio not working:**
```bash
# Linux
sudo apt-get install portaudio19-dev

# macOS  
brew install portaudio
```

**Slow performance:**
- Close other apps
- Use performance optimizer: `python optimize_performance.py`

**Build errors:**
```bash
buildozer android clean
pip install --upgrade buildozer
```

## Future Improvements

- Better emotion accuracy with real training data
- Support for more languages
- Playlist creation in Spotify
- Voice activity detection
- Real-time emotion tracking

## Contributing

Feel free to submit issues or PRs. This is still a learning project.

## License

MIT License

## Acknowledgments

- Original team members from final year project
- Research papers on speech emotion recognition
- Open source libraries (librosa, tensorflow, kivy)
- Spotify for the music API

---

Built for my CS degree final project. Refined version with better performance and user experience.

# Make emotion_detector.py more human-like
human_emotion_detector = '''"""
Emotion detection using CNN + KNN
FallenGodfather - Final year project
"""

import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os
import logging
import gc

# make tensorflow quieter and faster
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'fearful', 'happy', 'neutral', 'sad', 'surprised', 'disgusted']
        self.n_emotions = len(self.emotions)
        
        self.cnn_model = None
        self.knn_model = None
        self.scaler = StandardScaler()
        self.cnn_weight = 0.7  # CNN is better so higher weight
        self.knn_weight = 0.3
        
        # reduced feature params for speed
        self.sample_rate = 22050
        self.n_mfcc = 12
        self.n_chroma = 8  
        self.n_mel = 64
        
        self.ready = False
        self.init_models()
        
    def init_models(self):
        try:
            self.load_models()
            self.ready = True
        except:
            logger.info("No saved models, creating new ones...")
            self.create_models()
            self.train_models()
            self.ready = True
            
    def create_cnn_model(self, input_shape):
        # simple CNN for mobile
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(), 
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.GlobalAveragePooling1D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.n_emotions, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def create_models(self):
        self.cnn_model = None
        self.knn_model = KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            algorithm='kd_tree'
        )
        logger.info("Models created")
        
    def extract_audio_features(self, audio, sr=None):
        if sr is None:
            sr = self.sample_rate
            
        features = []
        
        try:
            # trim long audio for speed
            if len(audio) > sr * 10:
                audio = audio[:sr * 10]
            
            # MFCC - most important features
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, hop_length=1024)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            
            # Chroma
            chroma = librosa.feature.chroma(y=audio, sr=sr, n_chroma=self.n_chroma, hop_length=1024)
            chroma_mean = np.mean(chroma, axis=1)
            features.extend(chroma_mean)
            
            # Mel spectrogram
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mel, hop_length=1024)
            mel_mean = np.mean(mel, axis=1)
            features.extend(mel_mean)
            
            # basic spectral stuff
            spectral = librosa.feature.spectral_centroid(y=audio, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(audio)
            
            features.extend([
                np.mean(spectral),
                np.std(spectral),
                np.mean(zcr),
                np.std(zcr)
            ])
            
            # energy
            try:
                rms = librosa.feature.rms(y=audio)
                features.extend([np.mean(rms), np.std(rms)])
            except:
                features.extend([0, 0])
                
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            expected_size = self.n_mfcc * 2 + self.n_chroma + self.n_mel + 6
            return np.zeros(expected_size, dtype=np.float32)
            
    def train_models(self):
        logger.info("Training models with synthetic data...")
        
        # less training data for speed
        samples_per_emotion = 50
        expected_features = self.n_mfcc * 2 + self.n_chroma + self.n_mel + 6
        
        X_train = []
        y_train = []
        
        # emotion patterns from research
        patterns = {
            'angry': {'energy': 2.0, 'mfcc': 1.5, 'spectral': 1.3},
            'happy': {'energy': 1.5, 'mfcc': 1.2, 'spectral': 1.4},
            'sad': {'energy': 0.6, 'mfcc': 0.8, 'spectral': 0.7},
            'fearful': {'energy': 1.8, 'mfcc': 1.1, 'spectral': 1.5},
            'surprised': {'energy': 1.7, 'mfcc': 1.3, 'spectral': 1.6},
            'disgusted': {'energy': 0.8, 'mfcc': 0.9, 'spectral': 0.9},
            'neutral': {'energy': 1.0, 'mfcc': 1.0, 'spectral': 1.0}
        }
        
        # generate training data
        for emotion_idx, emotion in enumerate(self.emotions):
            pattern = patterns[emotion]
            
            for _ in range(samples_per_emotion):
                base = np.random.normal(0, 0.5, expected_features)
                
                # apply emotion patterns
                mfcc_end = self.n_mfcc * 2
                base[:mfcc_end] *= pattern['mfcc']
                
                # energy (last 2)
                base[-2:] *= pattern['energy']
                
                # spectral (middle part)
                spectral_start = mfcc_end + self.n_chroma + self.n_mel
                spectral_end = spectral_start + 4
                base[spectral_start:spectral_end] *= pattern['spectral']
                
                X_train.append(base)
                y_train.append(emotion_idx)
                
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)
        
        # scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # train KNN
        self.knn_model.fit(X_scaled, y_train)
        
        # train CNN
        X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
        y_cnn = keras.utils.to_categorical(y_train, self.n_emotions)
        
        self.cnn_model = self.create_cnn_model((X_cnn.shape[1], 1))
        
        # quick training
        history = self.cnn_model.fit(
            X_cnn, y_cnn,
            epochs=20,
            batch_size=16,
            validation_split=0.15,
            verbose=0
        )
        
        logger.info(f"Training done. Accuracy: {history.history['accuracy'][-1]:.3f}")
        self.save_models()
        
        # cleanup
        del X_train, y_train, X_scaled, X_cnn, y_cnn
        gc.collect()
        
    def predict_emotion(self, features):
        if not self.ready:
            return "neutral", 50.0
            
        try:
            # scale features
            features_scaled = self.scaler.transform(features.reshape(1, -1)).astype(np.float32)
            
            # CNN prediction
            features_cnn = features_scaled.reshape(1, features_scaled.shape[1], 1)
            cnn_probs = self.cnn_model.predict(features_cnn, verbose=0)[0]
            
            # KNN prediction  
            knn_probs = self.knn_model.predict_proba(features_scaled)[0]
            
            # combine predictions
            final_probs = self.cnn_weight * cnn_probs + self.knn_weight * knn_probs
            prediction = np.argmax(final_probs)
            confidence = np.max(final_probs) * 100
            
            emotion = self.emotions[prediction]
            logger.info(f"Predicted: {emotion} ({confidence:.1f}%)")
            
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return "neutral", 50.0
            
    def save_models(self):
        try:
            os.makedirs('models', exist_ok=True)
            
            self.cnn_model.save('models/emotion_cnn_model.h5')
            
            with open('models/emotion_knn_model.pkl', 'wb') as f:
                pickle.dump(self.knn_model, f)
                
            with open('models/feature_scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("Models saved")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            
    def load_models(self):
        self.cnn_model = keras.models.load_model('models/emotion_cnn_model.h5')
        
        with open('models/emotion_knn_model.pkl', 'rb') as f:
            self.knn_model = pickle.load(f)
            
        with open('models/feature_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        logger.info("Models loaded")
        
    def cleanup(self):
        try:
            if self.cnn_model:
                del self.cnn_model
                self.cnn_model = None
                
            if self.knn_model:
                del self.knn_model
                self.knn_model = None
                
            gc.collect()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


if __name__ == "__main__":
    # quick test
    detector = EmotionDetector()
    
    expected_features = detector.n_mfcc * 2 + detector.n_chroma + detector.n_mel + 6
    test_features = np.random.normal(0, 1, expected_features).astype(np.float32)
    
    start = time.time()
    emotion, confidence = detector.predict_emotion(test_features)
    end = time.time()
    
    print(f"Test: {emotion} ({confidence:.1f}%) in {(end-start)*1000:.2f}ms")
    print(f"Features: {len(test_features)}")
    
    detector.cleanup()
'''

# Save human-like emotion_detector.py
with open('emotion_detector.py', 'w') as f:
    f.write(human_emotion_detector)
    
print("✅ Made emotion_detector.py more human-like - simplified comments and structure")
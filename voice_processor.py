"""
Voice recording and processing
FallenGodfather
"""

import pyaudio
import numpy as np
import threading
import queue
import time
import librosa
import logging
import gc
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self):
        # audio settings
        self.sample_rate = 22050
        self.chunk_size = 2048
        self.channels = 1
        self.format = pyaudio.paInt16

        self.recording = False
        self.audio_queue = queue.Queue(maxsize=200)
        self.audio_buffer = deque(maxlen=1000)

        self.audio = None
        self.stream = None

        self.chunk_count = 0

        self.init_audio()

    def init_audio(self):
        try:
            self.audio = pyaudio.PyAudio()
            logger.info("Audio initialized")
            self.find_best_device()

        except Exception as e:
            logger.error(f"Audio init failed: {e}")
            self.audio = None

    def find_best_device(self):
        if not self.audio:
            return

        try:
            default = self.audio.get_default_input_device_info()
            self.device_index = default['index']
            logger.info(f"Using: {default['name']}")

        except Exception as e:
            logger.error(f"Device error: {e}")
            self.device_index = None

    def start_recording(self):
        if self.recording:
            return False

        if not self.audio:
            return False

        try:
            # clear buffers
            self.audio_buffer.clear()
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break

            # setup stream
            kwargs = {
                'format': self.format,
                'channels': self.channels,
                'rate': self.sample_rate,
                'input': True,
                'frames_per_buffer': self.chunk_size,
                'stream_callback': self.audio_callback
            }

            if self.device_index:
                kwargs['input_device_index'] = self.device_index

            self.stream = self.audio.open(**kwargs)

            self.recording = True
            self.stream.start_stream()

            self.chunk_count = 0

            logger.info("Recording started")
            return True

        except Exception as e:
            logger.error(f"Start failed: {e}")
            return False

    def stop_recording(self):
        if not self.recording:
            return

        try:
            self.recording = False

            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            logger.info(f"Stopped after {self.chunk_count} chunks")

        except Exception as e:
            logger.error(f"Stop error: {e}")

    def audio_callback(self, data, frame_count, time_info, status):
        if self.recording:
            try:
                audio_chunk = np.frombuffer(data, dtype=np.int16)

                if not self.audio_queue.full():
                    self.audio_queue.put_nowait(audio_chunk)

                self.chunk_count += 1

            except Exception as e:
                logger.error(f"Callback error: {e}")

        return (data, pyaudio.paContinue)

    def record_audio(self, duration=20.0):
        if not self.start_recording():
            return None

        try:
            self.audio_buffer.clear()

            start_time = time.time()
            chunks = []

            logger.info(f"Recording for {duration}s...")

            while time.time() - start_time < duration and self.recording:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    chunks.append(chunk)

                    # don't collect too much
                    if len(chunks) > 500:
                        break

                except queue.Empty:
                    continue

            self.stop_recording()

            if not chunks:
                logger.warning("No audio recorded")
                return None

            logger.info(f"Got {len(chunks)} chunks")
            audio = np.concatenate(chunks)

            # convert and normalize
            audio = audio.astype(np.float32) / 32768.0

            # check if too quiet
            if np.max(np.abs(audio)) < 0.001:
                logger.warning("Audio too quiet")
                return None

            actual_duration = len(audio) / self.sample_rate
            logger.info(f"Recorded {actual_duration:.2f}s")

            del chunks
            gc.collect()

            return audio

        except Exception as e:
            logger.error(f"Record error: {e}")
            self.stop_recording()
            return None

    def extract_features(self, audio):
        try:
            if audio is None or len(audio) == 0:
                return None

            # clean up audio first
            audio = self.clean_audio(audio)

            # use emotion detector for features
            from emotion_detector import EmotionDetector

            detector = EmotionDetector()
            features = detector.extract_audio_features(audio, self.sample_rate)

            if features is not None:
                logger.info(f"Extracted {len(features)} features")

            return features

        except Exception as e:
            logger.error(f"Feature error: {e}")
            return None

    def clean_audio(self, audio):
        try:
            # remove silence
            audio_clean, _ = librosa.effects.trim(
                audio, 
                top_db=15,
                frame_length=2048,
                hop_length=512
            )

            # normalize
            if np.max(np.abs(audio_clean)) > 0:
                audio_clean = audio_clean / np.max(np.abs(audio_clean))

            # pre-emphasis if possible
            try:
                audio_clean = librosa.effects.preemphasis(audio_clean, coef=0.95)
            except:
                pass

            return audio_clean

        except Exception as e:
            logger.error(f"Clean error: {e}")
            return audio

    def check_speech(self, audio, threshold=0.005):
        try:
            # quick energy check
            rms = np.sqrt(np.mean(audio ** 2))

            # zero crossings
            zcr = np.sum(np.diff(np.signbit(audio))) / len(audio)

            has_speech = rms > threshold and zcr > 0.005

            return has_speech

        except Exception as e:
            logger.error(f"Speech check error: {e}")
            return False

    def get_level(self, audio):
        try:
            if audio is None or len(audio) == 0:
                return 0.0

            rms = np.sqrt(np.mean(audio ** 2))

            if rms > 0:
                db = 20 * np.log10(rms)
                level = max(0, min(100, (db + 60) * 100 / 60))
            else:
                level = 0.0

            return level

        except:
            return 0.0

    def get_info(self):
        return {
            'recording': self.recording,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'queue_size': self.audio_queue.qsize(),
            'chunks': self.chunk_count
        }

    def cleanup(self):
        try:
            self.stop_recording()

            self.audio_buffer.clear()
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except:
                    break

            if self.audio:
                self.audio.terminate()
                self.audio = None

            gc.collect()
            logger.info("Cleanup done")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    # quick test
    processor = VoiceProcessor()

    print("Testing voice processor...")
    print("Recording 5 seconds...")

    start = time.time()
    audio = processor.record_audio(duration=5.0)
    end = time.time()

    if audio is not None:
        print(f"Got {len(audio)} samples in {end-start:.2f}s")

        # test features
        start = time.time()
        features = processor.extract_features(audio)
        end = time.time()

        if features is not None:
            print(f"Features: {len(features)} in {(end-start)*1000:.2f}ms")

    processor.cleanup()
    print("Test done!")

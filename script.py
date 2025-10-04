# Make main.py more human-like
human_main_py = '''"""
Voice Emotion AI - Final Year Project
FallenGodfather
"""

import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock
from kivy.utils import get_color_from_hex
from kivy.core.window import Window
from kivy.uix.popup import Popup
import threading
import time
import gc
import importlib

kivy.require('2.0.0')

class MainApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.padding = 20
        self.spacing = 15
        
        # lazy load components to avoid startup lag
        self._detector = None
        self._spotify = None
        self._voice = None
        
        self.recording = False
        self.current_emotion = None
        self.recording_time = 20  # 20 seconds max
        self.stop_timer = None
        
        self.setup_ui()
        Window.clearcolor = get_color_from_hex('#f8f9fa')
        gc.collect()  # cleanup
        
    @property
    def detector(self):
        if self._detector is None:
            emotion_module = importlib.import_module('emotion_detector')
            self._detector = emotion_module.EmotionDetector()
        return self._detector
    
    @property
    def spotify(self):
        if self._spotify is None:
            spotify_module = importlib.import_module('spotify_integration')
            self._spotify = spotify_module.SpotifyIntegration()
        return self._spotify
    
    @property
    def voice(self):
        if self._voice is None:
            voice_module = importlib.import_module('voice_processor')
            self._voice = voice_module.VoiceProcessor()
        return self._voice
        
    def setup_ui(self):
        # title
        title = Label(
            text='Voice Emotion AI\\nby FallenGodfather',
            font_size='22sp',
            size_hint=(1, 0.12),
            color=get_color_from_hex('#2c3e50'),
            halign='center',
            bold=True
        )
        title.bind(size=title.setter('text_size'))
        self.add_widget(title)
        
        # status
        self.status = Label(
            text='Ready - Tap mic to start (20 seconds)',
            font_size='15sp',
            size_hint=(1, 0.08),
            color=get_color_from_hex('#34495e'),
            halign='center'
        )
        self.status.bind(size=self.status.setter('text_size'))
        self.add_widget(self.status)
        
        # emotion display
        self.emotion_display = Label(
            text='😊 Tap the mic to start',
            font_size='28sp',
            size_hint=(1, 0.18),
            color=get_color_from_hex('#e74c3c'),
            halign='center',
            bold=True
        )
        self.emotion_display.bind(size=self.emotion_display.setter('text_size'))
        self.add_widget(self.emotion_display)
        
        # progress
        self.progress = ProgressBar(max=100, value=0, size_hint=(1, 0.04))
        self.add_widget(self.progress)
        
        # timer
        self.timer = Label(
            text='',
            font_size='14sp',
            size_hint=(1, 0.06),
            color=get_color_from_hex('#7f8c8d'),
            halign='center'
        )
        self.timer.bind(size=self.timer.setter('text_size'))
        self.add_widget(self.timer)
        
        # buttons
        btn_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.14), spacing=10)
        
        self.record_btn = Button(
            text='🎤 Start Recording',
            font_size='16sp',
            background_color=get_color_from_hex('#27ae60'),
            size_hint=(0.5, 1),
            bold=True
        )
        self.record_btn.bind(on_press=self.toggle_recording)
        btn_layout.add_widget(self.record_btn)
        
        self.music_btn = Button(
            text='🎵 Get Music',
            font_size='16sp',
            background_color=get_color_from_hex('#1db954'),
            size_hint=(0.5, 1),
            bold=True,
            disabled=True
        )
        self.music_btn.bind(on_press=self.get_music)
        btn_layout.add_widget(self.music_btn)
        
        self.add_widget(btn_layout)
        
        # questions area
        self.question = Label(
            text='',
            font_size='13sp',
            size_hint=(1, 0.16),
            color=get_color_from_hex('#7f8c8d'),
            halign='center',
            valign='middle'
        )
        self.question.bind(size=self.question.setter('text_size'))
        self.add_widget(self.question)
        
        # footer
        footer = Label(
            text='AI emotion detection + music recommendations\\nFinal Year Project - Enhanced',
            font_size='11sp',
            size_hint=(1, 0.1),
            color=get_color_from_hex('#95a5a6'),
            halign='center'
        )
        footer.bind(size=footer.setter('text_size'))
        self.add_widget(footer)
        
    def toggle_recording(self, btn):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def start_recording(self):
        self.recording = True
        self.record_btn.text = '⏹️ Recording...'
        self.record_btn.background_color = get_color_from_hex('#e74c3c')
        self.status.text = 'Recording... speak naturally'
        self.music_btn.disabled = True
        self.progress.value = 0
        
        # pick a random question
        questions = [
            "Tell me how you're feeling today...",
            "What's your mood right now?",
            "Share what's on your mind...",
            "How has your day been?",
            "Describe your current emotions..."
        ]
        
        import random
        self.question.text = f"💭 {random.choice(questions)}"
        
        # auto stop after 20 seconds
        self.stop_timer = Clock.schedule_once(self.auto_stop, self.recording_time)
        
        # start recording thread
        thread = threading.Thread(target=self.record_and_analyze, daemon=True)
        thread.start()
        
        # start UI updates
        self.start_time = time.time()
        Clock.schedule_interval(self.update_ui, 0.1)
        
    def auto_stop(self, dt):
        if self.recording:
            self.stop_recording()
            
    def stop_recording(self):
        self.recording = False
        self.voice.stop_recording()
        
        if self.stop_timer:
            self.stop_timer.cancel()
            self.stop_timer = None
            
        Clock.unschedule(self.update_ui)
        
    def record_and_analyze(self):
        try:
            audio = self.voice.record_audio(duration=self.recording_time)
            
            if audio is not None and len(audio) > 0:
                features = self.voice.extract_features(audio)
                
                if features is not None:
                    emotion, confidence = self.detector.predict_emotion(features)
                    Clock.schedule_once(lambda dt: self.show_result(emotion, confidence))
                else:
                    Clock.schedule_once(lambda dt: self.show_error("Couldn't analyze voice"))
            else:
                Clock.schedule_once(lambda dt: self.show_error("No voice detected"))
                
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_error(f"Error: {str(e)}"))
        finally:
            gc.collect()
            
    def update_ui(self, dt):
        if self.recording:
            elapsed = time.time() - self.start_time
            progress = (elapsed / self.recording_time) * 100
            remaining = max(0, self.recording_time - elapsed)
            
            self.progress.value = min(progress, 100)
            self.timer.text = f"Time left: {remaining:.1f}s"
            
            if progress >= 100:
                Clock.unschedule(self.update_ui)
                
    def show_result(self, emotion, confidence):
        self.recording = False
        self.record_btn.text = '🎤 Start Recording'
        self.record_btn.background_color = get_color_from_hex('#27ae60')
        
        emojis = {
            'happy': '😊', 'sad': '😢', 'angry': '😠',
            'fearful': '😰', 'surprised': '😲', 'disgusted': '🤢', 'neutral': '😐'
        }
        
        emoji = emojis.get(emotion.lower(), '🤔')
        self.current_emotion = emotion
        
        self.emotion_display.text = f'{emoji} {emotion.title()}'
        self.status.text = f'Detected: {confidence:.1f}% confidence'
        self.question.text = f'You seem to be feeling {emotion}. Want some music?'
        self.timer.text = 'Done!'
        
        self.music_btn.disabled = False
        self.progress.value = 100
        gc.collect()
        
    def get_music(self, btn):
        if self.current_emotion:
            self.music_btn.text = '🔄 Loading...'
            self.music_btn.disabled = True
            
            thread = threading.Thread(
                target=self.get_recommendations,
                args=(self.current_emotion,),
                daemon=True
            )
            thread.start()
            
    def get_recommendations(self, emotion):
        try:
            recs = self.spotify.get_recommendations_by_emotion(emotion)
            Clock.schedule_once(lambda dt: self.show_music(recs))
        except Exception as e:
            Clock.schedule_once(lambda dt: self.music_error(str(e)))
            
    def show_music(self, recs):
        self.music_btn.text = '🎵 Get Music'
        self.music_btn.disabled = False
        
        if recs:
            self.music_popup(recs)
        else:
            self.spotify.open_spotify_with_mood(self.current_emotion)
            
    def music_error(self, error):
        self.music_btn.text = '🎵 Get Music'
        self.music_btn.disabled = False
        self.show_error(f"Music error: {error}")
        
    def music_popup(self, recs):
        content = BoxLayout(orientation='vertical', spacing=8, padding=8)
        
        title = Label(
            text=f'🎵 Music for {self.current_emotion} mood',
            font_size='16sp',
            size_hint=(1, 0.15),
            bold=True
        )
        content.add_widget(title)
        
        # show top 3 songs
        for track in recs[:3]:
            btn = Button(
                text=f"{track['name'][:25]}... - {track['artist'][:15]}...",
                font_size='13sp',
                size_hint=(1, 0.18),
                background_color=get_color_from_hex('#1db954')
            )
            btn.bind(on_press=lambda x, url=track['spotify_url']: self.spotify.open_spotify_track(url))
            content.add_widget(btn)
            
        # bottom buttons
        btn_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.15), spacing=5)
        
        more_btn = Button(text='More', font_size='14sp', background_color=get_color_from_hex('#3498db'))
        close_btn = Button(text='Close', font_size='14sp', background_color=get_color_from_hex('#95a5a6'))
        
        popup = Popup(title='Music Recommendations', content=content, size_hint=(0.85, 0.65))
        
        more_btn.bind(on_press=lambda x: self.open_more_music(popup))
        close_btn.bind(on_press=popup.dismiss)
        
        btn_row.add_widget(more_btn)
        btn_row.add_widget(close_btn)
        content.add_widget(btn_row)
        
        popup.open()
        
    def open_more_music(self, popup):
        popup.dismiss()
        self.spotify.open_spotify_with_mood(self.current_emotion)
        
    def show_error(self, msg):
        self.status.text = f'Error: {msg}'
        self.emotion_display.text = '❌ Failed'
        self.timer.text = 'Try again'
        self.recording = False
        self.record_btn.text = '🎤 Start Recording'
        self.record_btn.background_color = get_color_from_hex('#27ae60')
        
        # reset after 3 seconds
        Clock.schedule_once(self.reset_ui, 3.0)
        
    def reset_ui(self, dt):
        self.status.text = 'Ready - Tap mic to start (20 seconds)'
        self.emotion_display.text = '😊 Tap the mic to start'
        self.question.text = ''
        self.timer.text = ''
        self.progress.value = 0


class VoiceEmotionApp(App):
    def build(self):
        return MainApp()
        
    def on_stop(self):
        # cleanup when app closes
        if hasattr(self.root, '_voice') and self.root._voice:
            self.root._voice.cleanup()
        gc.collect()


if __name__ == '__main__':
    VoiceEmotionApp().run()
'''

# Save human-like main.py
with open('main.py', 'w') as f:
    f.write(human_main_py)
    
print("✅ Made main.py more human-like - removed verbose comments and AI patterns")
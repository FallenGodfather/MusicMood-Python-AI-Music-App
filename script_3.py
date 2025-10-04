# Make spotify_integration.py more human-like
human_spotify = '''"""
Spotify integration for music recommendations
FallenGodfather
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
import urllib.parse
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyIntegration:
    def __init__(self):
        # need to replace with real credentials
        self.client_id = "YOUR_SPOTIFY_CLIENT_ID"
        self.client_secret = "YOUR_SPOTIFY_CLIENT_SECRET"
        
        self.spotify = None
        
        # music params for different emotions
        self.mood_settings = {
            'happy': {
                'valence': 0.8,
                'energy': 0.8,
                'danceability': 0.7,
                'genres': ['pop', 'dance', 'funk', 'disco']
            },
            'sad': {
                'valence': 0.2,
                'energy': 0.3,
                'danceability': 0.3,
                'genres': ['indie', 'folk', 'acoustic', 'ambient']
            },
            'angry': {
                'valence': 0.3,
                'energy': 0.9,
                'danceability': 0.6,
                'genres': ['rock', 'metal', 'punk', 'alternative']
            },
            'fearful': {
                'valence': 0.4,
                'energy': 0.4,
                'danceability': 0.2,
                'genres': ['ambient', 'classical', 'new-age']
            },
            'surprised': {
                'valence': 0.6,
                'energy': 0.7,
                'danceability': 0.6,
                'genres': ['electronic', 'experimental', 'synthpop']
            },
            'disgusted': {
                'valence': 0.3,
                'energy': 0.5,
                'danceability': 0.4,
                'genres': ['grunge', 'alternative', 'indie-rock']
            },
            'neutral': {
                'valence': 0.5,
                'energy': 0.5,
                'danceability': 0.5,
                'genres': ['pop', 'indie', 'alternative', 'chill']
            }
        }
        
        self.init_spotify()
        
    def init_spotify(self):
        try:
            if self.client_id != "YOUR_SPOTIFY_CLIENT_ID":
                auth = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.spotify = spotipy.Spotify(client_credentials_manager=auth)
                logger.info("Spotify connected")
            else:
                logger.warning("Need to set Spotify credentials")
                
        except Exception as e:
            logger.error(f"Spotify init failed: {e}")
            
    def get_recommendations_by_emotion(self, emotion, limit=10):
        if not self.spotify:
            return self.fallback_songs(emotion)
            
        try:
            emotion = emotion.lower()
            if emotion not in self.mood_settings:
                emotion = 'neutral'
                
            settings = self.mood_settings[emotion]
            
            # get recommendations from spotify
            recs = self.spotify.recommendations(
                seed_genres=settings['genres'][:5],
                limit=limit,
                target_valence=settings['valence'],
                target_energy=settings['energy'],
                target_danceability=settings['danceability'],
                market='US'
            )
            
            # format results
            songs = []
            for track in recs['tracks']:
                song = {
                    'name': track['name'],
                    'artist': ', '.join([artist['name'] for artist in track['artists']]),
                    'album': track['album']['name'],
                    'spotify_url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url'],
                    'popularity': track['popularity']
                }
                songs.append(song)
                
            logger.info(f"Got {len(songs)} songs for {emotion}")
            return songs
            
        except Exception as e:
            logger.error(f"Recommendation error: {e}")
            return self.fallback_songs(emotion)
            
    def fallback_songs(self, emotion):
        # backup songs when API doesn't work
        songs = {
            'happy': [
                {'name': 'Happy', 'artist': 'Pharrell Williams', 'search': 'Happy Pharrell Williams'},
                {'name': 'Good as Hell', 'artist': 'Lizzo', 'search': 'Good as Hell Lizzo'},
                {'name': "Can't Stop the Feeling", 'artist': 'Justin Timberlake', 'search': 'Can\\'t Stop the Feeling Justin Timberlake'}
            ],
            'sad': [
                {'name': 'Someone Like You', 'artist': 'Adele', 'search': 'Someone Like You Adele'},
                {'name': 'Hurt', 'artist': 'Johnny Cash', 'search': 'Hurt Johnny Cash'},
                {'name': 'Mad World', 'artist': 'Gary Jules', 'search': 'Mad World Gary Jules'}
            ],
            'angry': [
                {'name': 'Break Stuff', 'artist': 'Limp Bizkit', 'search': 'Break Stuff Limp Bizkit'},
                {'name': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'search': 'Killing in the Name Rage'},
                {'name': 'Bodies', 'artist': 'Drowning Pool', 'search': 'Bodies Drowning Pool'}
            ],
            'fearful': [
                {'name': 'Clair de Lune', 'artist': 'Debussy', 'search': 'Clair de Lune Debussy'},
                {'name': 'Weightless', 'artist': 'Marconi Union', 'search': 'Weightless Marconi Union'},
                {'name': 'Aqueous Transmission', 'artist': 'Incubus', 'search': 'Aqueous Transmission Incubus'}
            ],
            'surprised': [
                {'name': "What's Up?", 'artist': '4 Non Blondes', 'search': 'What\\'s Up 4 Non Blondes'},
                {'name': 'Bohemian Rhapsody', 'artist': 'Queen', 'search': 'Bohemian Rhapsody Queen'},
                {'name': 'Mr. Blue Sky', 'artist': 'ELO', 'search': 'Mr Blue Sky ELO'}
            ],
            'disgusted': [
                {'name': 'Smells Like Teen Spirit', 'artist': 'Nirvana', 'search': 'Smells Like Teen Spirit Nirvana'},
                {'name': 'Toxicity', 'artist': 'System of a Down', 'search': 'Toxicity System Down'},
                {'name': 'Crawling', 'artist': 'Linkin Park', 'search': 'Crawling Linkin Park'}
            ],
            'neutral': [
                {'name': 'Shape of You', 'artist': 'Ed Sheeran', 'search': 'Shape of You Ed Sheeran'},
                {'name': 'Blinding Lights', 'artist': 'The Weeknd', 'search': 'Blinding Lights Weeknd'},
                {'name': 'Levitating', 'artist': 'Dua Lipa', 'search': 'Levitating Dua Lipa'}
            ]
        }
        
        emotion = emotion.lower()
        tracks = songs.get(emotion, songs['neutral'])
        
        # format same as API results
        result = []
        for track in tracks:
            formatted = {
                'name': track['name'],
                'artist': track['artist'],
                'album': 'Unknown',
                'spotify_url': f"https://open.spotify.com/search/{urllib.parse.quote(track['search'])}",
                'preview_url': None,
                'popularity': 80
            }
            result.append(formatted)
            
        return result
        
    def open_spotify_with_mood(self, emotion):
        try:
            emotion = emotion.lower()
            searches = {
                'happy': 'happy upbeat music',
                'sad': 'sad emotional music',
                'angry': 'angry rock metal music',
                'fearful': 'calm relaxing music',
                'surprised': 'energetic dance music',
                'disgusted': 'alternative grunge music',
                'neutral': 'popular music'
            }
            
            query = searches.get(emotion, 'music')
            encoded = urllib.parse.quote(query)
            
            # try app first, then web
            app_url = f"spotify:search:{encoded}"
            web_url = f"https://open.spotify.com/search/{encoded}"
            
            try:
                webbrowser.open(app_url)
                logger.info(f"Opened Spotify app: {query}")
            except:
                webbrowser.open(web_url)
                logger.info(f"Opened Spotify web: {query}")
                
        except Exception as e:
            logger.error(f"Open Spotify error: {e}")
            
    def open_spotify_track(self, url):
        try:
            webbrowser.open(url)
            logger.info(f"Opened track: {url}")
        except Exception as e:
            logger.error(f"Track open error: {e}")


if __name__ == "__main__":
    # test
    spotify = SpotifyIntegration()
    
    recs = spotify.get_recommendations_by_emotion('happy')
    print(f"Got {len(recs)} recommendations for happy mood")
    
    for i, track in enumerate(recs[:3]):
        print(f"{i+1}. {track['name']} - {track['artist']}")
'''

# Save human-like spotify_integration.py
with open('spotify_integration.py', 'w') as f:
    f.write(human_spotify)
    
print("✅ Made spotify_integration.py more human-like - simplified and casual")
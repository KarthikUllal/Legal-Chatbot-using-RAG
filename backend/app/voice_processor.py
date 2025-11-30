# backend/app/voice_processor.py
import speech_recognition as sr
import logging
from fastapi import HTTPException
import io
import base64
import wave
import tempfile
import os

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.supported_languages = {
            'en': 'en-US',
            'hi': 'hi-IN', 
            'kn': 'kn-IN',
            'ta': 'ta-IN',
            'te': 'te-IN',
            'mr': 'mr-IN'
        }
    
    def process_audio(self, audio_data: str, language: str = 'en') -> str:
        """
        Process base64 audio data and convert to text
        """
        try:
            logger.info(f"Processing voice input in language: {language}")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name
            
            try:
                # Use speech recognition
                with sr.AudioFile(temp_audio_path) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.record(source)
                
                # Convert speech to text
                lang_code = self.supported_languages.get(language, 'en-US')
                
                text = self.recognizer.recognize_google(audio, language=lang_code)
                logger.info(f"Voice recognition successful: {text}")
                
                return text
                
            except sr.UnknownValueError:
                logger.warning("Speech recognition could not understand audio")
                return "Sorry, I couldn't understand the audio. Please try again."
                
            except sr.RequestError as e:
                logger.error(f"Speech recognition error: {e}")
                return "Speech recognition service is unavailable. Please try typing your question."
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            logger.error(f"Voice processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")

# Global instance
voice_processor = VoiceProcessor()
# backend/app/translation.py
import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class LegalTranslator:
    def __init__(self):
        self.supported_languages = {
            'en': 'English',
            'hi': 'Hindi',
            'kn': 'Kannada', 
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'bn': 'Bengali'
        }
        
        # Predefined legal terminology translations
        self.legal_terms = {
            'hi': {
                'section': 'धारा',
                'ipc': 'भारतीय दंड संहिता',
                'punishment': 'सजा',
                'cheating': 'मोस',
                'murder': 'हत्या',
                'consumer': 'उपभोक्ता',
                'rights': 'अधिकार',
                'cyber crime': 'साइबर अपराध',
                'domestic violence': 'घरेलू हिंसा'
            },
            'kn': {
                'section': 'ವಿಭಾಗ',
                'ipc': 'ಭಾರತೀಯ ದಂಡ ಸಂಹಿತೆ',
                'punishment': 'ಶಿಕ್ಷೆ',
                'cheating': 'ಮೋಸ',
                'murder': 'ಕೊಲೆ',
                'consumer': 'ಉಪಭೋಕ್ತ',
                'rights': 'ಹಕ್ಕುಗಳು',
                'cyber crime': 'ಸೈಬರ್ ಅಪರಾಧ',
                'domestic violence': 'ಕುಟುಂಬ ಹಿಂಸೆ'
            },
            'ta': {
                'section': 'பிரிவு',
                'ipc': 'இந்திய தண்டனை சட்டம்',
                'punishment': 'தண்டனை',
                'cheating': 'மோசடி',
                'murder': 'கொலை',
                'consumer': 'நுகர்வோர்',
                'rights': 'உரிமைகள்',
                'cyber crime': 'சைபர் குற்றம்',
                'domestic violence': 'குடும்ப வன்முறை'
            }
        }
    
    def translate_legal_response(self, english_response: str, target_language: str) -> str:
        """Translate English legal responses to target language"""
        if target_language == 'en':
            return english_response
            
        try:
            # First, replace legal terms with translated versions
            translated_text = english_response
            if target_language in self.legal_terms:
                for eng_term, regional_term in self.legal_terms[target_language].items():
                    translated_text = translated_text.replace(eng_term, regional_term)
                    translated_text = translated_text.replace(eng_term.title(), regional_term)
                    translated_text = translated_text.replace(eng_term.upper(), regional_term)
            
            # Then use Google Translate for the rest
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': 'en',
                'tl': target_language,
                'dt': 't',
                'q': translated_text
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                final_translation = ''.join([item[0] for item in result[0] if item[0]])
                logger.info(f"✅ Translated legal response to {self.supported_languages[target_language]}")
                return final_translation
            else:
                logger.warning(f"Translation failed, returning English")
                return english_response
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return english_response

# Global instance
translator = LegalTranslator()
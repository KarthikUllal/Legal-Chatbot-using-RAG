# test_translation.py
import requests
import json

class TranslationTester:
    def __init__(self):
        self.supported_languages = {
            'hi': 'Hindi',
            'kn': 'Kannada', 
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'bn': 'Bengali',
            'en': 'English'
        }
    
    def test_google_translate_free(self, text: str, target_lang: str, source_lang: str = 'en'):
        """Test FREE Google Translate API"""
        print(f"\n🔍 Testing: '{text}' -> {self.supported_languages[target_lang]}")
        
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            print(f"📡 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                translated_text = ''.join([item[0] for item in result[0] if item[0]])
                print(f"✅ SUCCESS: {translated_text}")
                return translated_text
            else:
                print(f"❌ FAILED: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"💥 ERROR: {e}")
            return None

def main():
    tester = TranslationTester()
    
    # Test phrases
    test_phrases = [
        "Hello, how are you?",
        "What is your name?",
        "I need help with legal matters",
        "What is the punishment for murder?",
        "Thank you very much"
    ]
    
    # Test languages
    test_languages = ['hi', 'kn', 'ta', 'te']
    
    print("🚀 STARTING MULTILINGUAL TRANSLATION TEST")
    print("=" * 50)
    
    for phrase in test_phrases:
        print(f"\n📝 Original: '{phrase}'")
        print("-" * 40)
        
        for lang in test_languages:
            tester.test_google_translate_free(phrase, lang)

if __name__ == "__main__":
    main()
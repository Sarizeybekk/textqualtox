import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Metin dilini algılayan sınıf.
    İstatistiksel yöntemlerle metinlerin dilini tespit eder.
    """

    def __init__(self):
        """Dil algılama sınıfını başlatır"""
        # Dil tanıma için tipik karakter ve kelime varlıkları
        self.language_profiles = {
            'tr': {
                'chars': 'abcçdefgğhıijklmnoöprsştuüvyz',
                'unique_chars': 'çğıöşü',
                'common_words': [
                    've', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'ben', 'sen', 'o',
                    'biz', 'siz', 'ama', 'ki', 'ya', 'çok', 'daha', 'en', 'ne', 'kadar',
                    'var', 'yok', 'mı', 'mi', 'mu', 'mü', 'gibi', 'olarak', 'çünkü',
                    'sonra', 'önce', 'nasıl', 'neden', 'evet', 'hayır', 'ise', 'veya'
                ]
            },
            'en': {
                'chars': 'abcdefghijklmnopqrstuvwxyz',
                'unique_chars': 'qwxz',
                'common_words': [
                    'the', 'and', 'a', 'to', 'of', 'in', 'is', 'you', 'that', 'it',
                    'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
                    'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
                    'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can',
                    'there', 'if', 'more', 'an', 'who'
                ]
            },
            'de': {
                'chars': 'abcdefghijklmnopqrstuvwxyzäöüß',
                'unique_chars': 'äöüß',
                'common_words': [
                    'der', 'die', 'das', 'und', 'in', 'zu', 'den', 'mit', 'auf', 'für',
                    'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es',
                    'von', 'sich', 'oder', 'so', 'zum', 'bei', 'eines', 'nur', 'am',
                    'werden', 'noch', 'wie', 'einer', 'aber', 'aus', 'wenn', 'doch'
                ]
            },
            'fr': {
                'chars': 'abcdefghijklmnopqrstuvwxyzàâçéèêëîïôùûü',
                'unique_chars': 'àâçéèêëîïôùûü',
                'common_words': [
                    'le', 'la', 'les', 'de', 'des', 'un', 'une', 'et', 'est', 'en',
                    'du', 'dans', 'qui', 'que', 'pour', 'pas', 'sur', 'ce', 'vous',
                    'avec', 'au', 'il', 'je', 'sont', 'mais', 'nous', 'si', 'plus',
                    'leur', 'par', 'ont', 'ou', 'comme', 'elle', 'tout', 'même'
                ]
            },
            'es': {
                'chars': 'abcdefghijklmnopqrstuvwxyzáéíóúüñ',
                'unique_chars': 'áéíóúüñ',
                'common_words': [
                    'el', 'la', 'los', 'las', 'de', 'del', 'un', 'una', 'unos', 'unas',
                    'y', 'e', 'o', 'u', 'que', 'en', 'a', 'con', 'por', 'para', 'es',
                    'son', 'al', 'lo', 'su', 'sus', 'se', 'mi', 'me', 'te', 'nos',
                    'como', 'pero', 'más', 'este', 'esta', 'esto'
                ]
            }
        }

        # Desteklenen diller
        self.supported_languages = {
            'tr': 'Türkçe',
            'en': 'İngilizce',
            'de': 'Almanca',
            'fr': 'Fransızca',
            'es': 'İspanyolca',
            'unknown': 'Bilinmeyen'
        }

        logger.info("Dil algılama modülü başlatıldı")

    def _clean_text(self, text):
        """
        Metni temizler

        Args:
            text: Temizlenecek metin

        Returns:
            str: Temizlenmiş metin
        """
        if not text:
            return ""

        # Küçük harfe çevir
        text = text.lower()

        # Sayıları ve özel karakterleri kaldır (dil karakterleri hariç)
        text = re.sub(r'[0-9]', '', text)
        text = re.sub(r'[^\w\s\u00C0-\u00FF\u0100-\u017F\u0400-\u04FF]', '', text)

        return text

    def _get_words(self, text):
        """
        Metinden kelimeleri çıkarır

        Args:
            text: Kelimesi çıkarılacak metin

        Returns:
            list: Kelimeler listesi
        """
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def _calculate_character_score(self, text, language):
        """
        Metindeki karakterlerin dile uygunluğunu hesaplar

        Args:
            text: Değerlendirilecek metin
            language: Dil kodu

        Returns:
            float: Karakter skoru (0-1 arası)
        """
        if not text:
            return 0.0

        profile = self.language_profiles.get(language, {})
        chars = profile.get('chars', '')
        unique_chars = profile.get('unique_chars', '')

        # Metin içindeki karakterleri say
        char_count = Counter(text.lower())

        # Dildeki karakterlerin metinde bulunma oranı
        total_chars = sum(char_count.values())
        if total_chars == 0:
            return 0.0

        matched_chars = sum(char_count.get(char, 0) for char in chars)
        char_ratio = matched_chars / total_chars

        # Dile özgü karakterlerin varlığını kontrol et
        unique_char_present = any(char in text.lower() for char in unique_chars)
        unique_bonus = 0.2 if unique_char_present else 0.0

        return min(1.0, char_ratio + unique_bonus)

    def _calculate_word_score(self, words, language):
        """
        Kelimelerin dile uygunluğunu hesaplar

        Args:
            words: Değerlendirilecek kelimeler listesi
            language: Dil kodu

        Returns:
            float: Kelime skoru (0-1 arası)
        """
        if not words:
            return 0.0

        common_words = self.language_profiles.get(language, {}).get('common_words', [])

        # Yaygın kelimelerin metinde bulunma sayısı
        matched_words = sum(1 for word in words if word in common_words)

        # Yaygın kelime oranı
        word_ratio = matched_words / min(len(words), 100)  # En fazla 100 kelime değerlendir

        return word_ratio

    def detect_language(self, text):
        """
        Metnin dilini tespit eder

        Args:
            text: Dili tespit edilecek metin

        Returns:
            dict: {
                'language_code': dil kodu (tr, en, vb),
                'language_name': dil adı,
                'confidence': güven skoru (0-1 arası),
                'scores': dil bazında skorlar
            }
        """
        if not text or len(text.strip()) < 5:
            return {
                'language_code': 'unknown',
                'language_name': self.supported_languages.get('unknown'),
                'confidence': 0.0,
                'scores': {}
            }

        try:
            clean_text = self._clean_text(text)
            words = self._get_words(clean_text)

            scores = {}

            # Her dil için skor hesapla
            for lang_code in self.language_profiles.keys():
                char_score = self._calculate_character_score(clean_text, lang_code)
                word_score = self._calculate_word_score(words, lang_code)

                # Ağırlıklı toplam (karakter:0.4, kelime:0.6)
                total_score = (char_score * 0.4) + (word_score * 0.6)
                scores[lang_code] = total_score

            # En yüksek skorlu dili bul
            if not scores:
                detected_lang = 'unknown'
                confidence = 0.0
            else:
                detected_lang = max(scores, key=scores.get)
                confidence = scores[detected_lang]

            # Eğer güven skoru çok düşükse "bilinmeyen" olarak işaretle
            if confidence < 0.15:
                detected_lang = 'unknown'
                confidence = 0.0

            return {
                'language_code': detected_lang,
                'language_name': self.supported_languages.get(detected_lang, self.supported_languages.get('unknown')),
                'confidence': confidence,
                'scores': scores
            }

        except Exception as e:
            logger.error(f"Dil algılama hatası: {str(e)}")
            return {
                'language_code': 'unknown',
                'language_name': self.supported_languages.get('unknown'),
                'confidence': 0.0,
                'scores': {}
            }

    def detect_languages_batch(self, texts):
        """
        Birden çok metnin dilini tespit eder

        Args:
            texts: Dilleri tespit edilecek metinler listesi

        Returns:
            list: Her metin için dil tespiti sonuçları
        """
        results = []

        for text in texts:
            result = self.detect_language(text)
            results.append(result)

        return results
import re
import logging
import string
from collections import Counter

logger = logging.getLogger(__name__)


class TextImprover:
    """
    Türkçe metinlerde iyileştirme ve öneriler sunan sınıf.
    Yazım hataları düzeltme, dilbilgisi önerileri ve okunabilirlik analizi yapar.
    """

    def __init__(self):
        """Metin iyileştirme sınıfını başlatır"""
        # Türkçe'de yaygın yazım hataları ve düzeltmeleri
        self.common_typos = {
            # Büyük küçük harf duyarsız olarak yazım hataları
            'bişey': 'bir şey',
            'herşey': 'her şey',
            'hiçbirşey': 'hiçbir şey',
            'birsey': 'bir şey',
            'hersey': 'her şey',
            'hicbir': 'hiçbir',
            'hicbirsey': 'hiçbir şey',
            'yalnız': 'yalnız',
            'bi': 'bir',
            'gelicek': 'gelecek',
            'gidiyom': 'gidiyorum',
            'yapıyom': 'yapıyorum',
            'biliyomusun': 'biliyor musun',
            'napıyorsun': 'ne yapıyorsun',
            'naber': 'ne haber',
            'bilmiyomki': 'bilmiyorum ki',
            'dicek': 'diyecek',
            'dicem': 'diyeceğim',
            'yicek': 'yiyecek',
            'yicem': 'yiyeceğim'
        }

        # Türkçe'de sık kullanılan doldurma kelimeleri
        self.filler_words = [
            'yani', 'işte', 'şey', 'falan', 'filan', 'hani', 'mesela',
            'aslında', 'ya', 'ki', 'de', 'da', 'ama', 'fakat', 'lakin',
            'gerçekten', 'kesinlikle', 'tabii', 'tabi', 'şimdi', 'sonra'
        ]

        # Türkçe cümle karmaşıklığını değerlendirmek için parametreler
        self.max_sentence_length = 25  # Kelime sayısı
        self.max_word_length = 6  # Ortalama kelime uzunluğu

        # Okunabilirlik için kullanılacak parametreler
        # Türkçe için uyarlanmış Flesch Reading Ease formülü
        self.readability_thresholds = {
            'çok_kolay': 90,
            'kolay': 80,
            'orta_kolay': 70,
            'orta': 60,
            'orta_zor': 50,
            'zor': 30,
            'çok_zor': 0
        }

        logger.info("Metin iyileştirme modülü başlatıldı")

    def fix_typos(self, text):
        """
        Metindeki yaygın yazım hatalarını düzeltir

        Args:
            text: Düzeltilecek metin

        Returns:
            dict: {
                'corrected_text': str,  # Düzeltilmiş metin
                'corrections': list,    # Yapılan düzeltmeler listesi
                'correction_count': int # Düzeltme sayısı
            }
        """
        corrected_text = text
        corrections = []

        # Önce metni kelimelere ayır
        words = re.findall(r'\b\w+\b', text.lower())

        # Her kelimeyi kontrol et
        for word in words:
            if word.lower() in self.common_typos:
                correct_word = self.common_typos[word.lower()]
                # Kelimenin metindeki tüm örneklerini düzelt
                # \b ile kelime sınırlarını belirt
                pattern = r'\b' + re.escape(word) + r'\b'
                corrected_text = re.sub(pattern, correct_word, corrected_text, flags=re.IGNORECASE)
                corrections.append(f"'{word}' -> '{correct_word}'")

        return {
            'corrected_text': corrected_text,
            'corrections': corrections,
            'correction_count': len(corrections)
        }

    def check_grammar(self, text):
        """
        Metindeki temel dilbilgisi sorunlarını kontrol eder

        Args:
            text: Kontrol edilecek metin

        Returns:
            dict: {
                'issues': list,  # Tespit edilen sorunlar listesi
                'suggestions': list, # Öneriler listesi
                'issue_count': int   # Sorun sayısı
            }
        """
        issues = []
        suggestions = []

        # Cümlelere ayır
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sentence in enumerate(sentences):
            # Büyük harfle başlama kontrolü
            if sentence and not sentence[0].isupper():
                issues.append(f"Cümle {i + 1}: Büyük harfle başlamıyor")
                suggestions.append(f"Cümle {i + 1}: '{sentence[0]}' -> '{sentence[0].upper()}'")

            # Cümle uzunluğu kontrolü
            words = sentence.split()
            if len(words) > self.max_sentence_length:
                issues.append(f"Cümle {i + 1}: Çok uzun ({len(words)} kelime)")
                suggestions.append(f"Cümle {i + 1}: Daha kısa cümlelere bölmeyi düşünün")

            # Noktalama kontrolü
            if i < len(sentences) - 1:  # Son cümle değilse
                if not text.find(sentence + ".") and not text.find(sentence + "!") and not text.find(sentence + "?"):
                    issues.append(f"Cümle {i + 1}: Noktalama işareti eksik olabilir")
                    suggestions.append(f"Cümle {i + 1}: Cümle sonuna uygun noktalama işareti ekleyin")

        return {
            'issues': issues,
            'suggestions': suggestions,
            'issue_count': len(issues)
        }

    def reduce_filler_words(self, text):
        """
        Metindeki doldurma kelimelerini tespit eder ve azaltma önerileri sunar

        Args:
            text: İyileştirilecek metin

        Returns:
            dict: {
                'filler_words': list,  # Bulunan doldurma kelimeleri
                'filler_count': int,   # Doldurma kelimesi sayısı
                'suggested_text': str  # Önerilen iyileştirilmiş metin
            }
        """
        # Metindeki kelimeleri bul
        words = re.findall(r'\b\w+\b', text.lower())

        # Doldurma kelimelerini ve sayılarını say
        filler_counter = Counter()
        for word in words:
            if word.lower() in self.filler_words:
                filler_counter[word.lower()] += 1

        # Metni kelime kelime işle ve fazla doldurma kelimelerini kaldır
        suggested_text = text
        for filler_word, count in filler_counter.items():
            if count > 1:  # Birden fazla geçiyorsa
                # Her bir örneği bul
                occurrences = list(re.finditer(r'\b' + re.escape(filler_word) + r'\b', suggested_text, re.IGNORECASE))

                # İlk geçtiği yer hariç diğerlerini kaldır
                for occurrence in occurrences[1:]:
                    start, end = occurrence.span()
                    # Eğer kelimenin önünde veya arkasında boşluk varsa, onu da kaldır
                    if start > 0 and suggested_text[start - 1] == ' ':
                        start -= 1
                    suggested_text = suggested_text[:start] + suggested_text[end:]

        return {
            'filler_words': list(filler_counter.keys()),
            'filler_count': sum(filler_counter.values()),
            'suggested_text': suggested_text
        }

    def calculate_readability(self, text):
        """
        Metnin okunabilirlik skorunu hesaplar (Türkçe'ye uyarlanmış Flesch Reading Ease)

        Args:
            text: Değerlendirilecek metin

        Returns:
            dict: {
                'score': float,        # Okunabilirlik skoru (0-100)
                'level': str,          # Okunabilirlik seviyesi
                'avg_sentence_length': float, # Ortalama cümle uzunluğu
                'avg_word_length': float      # Ortalama kelime uzunluğu
            }
        """
        # Cümleleri ve kelimeleri ayır
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        total_words = 0
        total_syllables = 0

        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            total_words += len(words)

            # Türkçe heceleri kabaca hesapla (sesli harf sayısı)
            for word in words:
                # Türkçedeki sesli harfler
                vowels = 'aeıioöuüAEIİOÖUÜ'
                syllable_count = sum(1 for char in word if char in vowels)
                # En az bir hece olmalı
                syllable_count = max(1, syllable_count)
                total_syllables += syllable_count

        # Hesaplamalar
        if len(sentences) == 0 or total_words == 0:
            return {
                'score': 100,  # Boş metin - en kolay
                'level': 'çok_kolay',
                'avg_sentence_length': 0,
                'avg_word_length': 0
            }

        avg_sentence_length = total_words / len(sentences)
        avg_syllables_per_word = total_syllables / total_words

        # Türkçe için uyarlanmış Flesch Reading Ease
        # (orijinal formül: 206.835 - 1.015 * ASL - 84.6 * ASW)
        # Türkçe için katsayılar ayarlandı
        readability_score = 206.835 - (1.3 * avg_sentence_length) - (60.0 * avg_syllables_per_word)

        # Skoru 0-100 aralığına sınırla
        readability_score = max(0, min(100, readability_score))

        # Seviyeyi belirle
        level = 'çok_zor'
        for threshold_level, threshold_value in sorted(self.readability_thresholds.items(), key=lambda x: x[1]):
            if readability_score >= threshold_value:
                level = threshold_level
                break

        return {
            'score': float(readability_score),
            'level': level,
            'avg_sentence_length': float(avg_sentence_length),
            'avg_word_length': float(avg_syllables_per_word)
        }

    def improve_text(self, text):
        """
        Metni kapsamlı şekilde analiz eder ve iyileştirme önerileri sunar

        Args:
            text: İyileştirilecek metin

        Returns:
            dict: Tüm iyileştirme analizlerini içeren sonuçlar
        """
        if not text or len(text.strip()) == 0:
            return {
                'corrected_text': text,
                'suggestions': [],
                'readability': {
                    'score': 100,
                    'level': 'çok_kolay'
                },
                'improvement_count': 0
            }

        try:
            # Yazım hatalarını düzelt
            typo_results = self.fix_typos(text)

            # Dilbilgisi kontrolü
            grammar_results = self.check_grammar(typo_results['corrected_text'])

            # Doldurma kelimelerini azalt
            filler_results = self.reduce_filler_words(typo_results['corrected_text'])

            # Okunabilirlik hesapla
            readability_results = self.calculate_readability(text)

            # Tüm önerileri birleştir
            all_suggestions = []
            all_suggestions.extend([f"Yazım düzeltmesi: {correction}" for correction in typo_results['corrections']])
            all_suggestions.extend(
                [f"Dilbilgisi önerisi: {suggestion}" for suggestion in grammar_results['suggestions']])

            if filler_results['filler_count'] > 0:
                all_suggestions.append(f"Doldurma kelimelerini azaltın: {', '.join(filler_results['filler_words'])}")

            # Okunabilirlik önerisi
            if readability_results['score'] < 60:  # Orta seviyenin altında ise
                if readability_results['avg_sentence_length'] > 15:
                    all_suggestions.append("Daha kısa cümleler kullanın (ortalama cümle uzunluğu yüksek)")
                if readability_results['avg_word_length'] > 2.5:
                    all_suggestions.append("Daha basit kelimeler kullanmayı deneyin (ortalama hece sayısı yüksek)")

            # Birleştirilmiş sonuç
            return {
                'original_text': text,
                'corrected_text': typo_results['corrected_text'],
                'improved_text': filler_results['suggested_text'],
                'suggestions': all_suggestions,
                'readability': {
                    'score': readability_results['score'],
                    'level': readability_results['level'],
                    'avg_sentence_length': readability_results['avg_sentence_length']
                },
                'improvement_count': len(all_suggestions)
            }

        except Exception as e:
            logger.error(f"Metin iyileştirme sırasında hata: {str(e)}")
            return {
                'original_text': text,
                'corrected_text': text,
                'improved_text': text,
                'suggestions': ["Metin analizi sırasında bir hata oluştu"],
                'readability': {
                    'score': 0,
                    'level': 'bilinmiyor'
                },
                'improvement_count': 0
            }
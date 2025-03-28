import numpy as np
from transformers import pipeline
import torch
import logging
import re
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QualityScorer:
    def __init__(self, quality_pipeline=None):
        """
        Metin kalitesi değerlendirme sınıfını başlatır.

        Args:
            quality_pipeline: Metin özetleme pipeline'ı
        """
        self.pipeline = quality_pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Temel kalite ölçütleri için yardımcı araçlar
        self.vectorizer = CountVectorizer(max_features=5000)

        # Türkçeye özgü doldurma (filler) kelimeleri
        self.turkish_filler_words = [
            'yani', 'işte', 'şey', 'falan', 'filan', 'hani', 'mesela', 'aslında',
            'ya', 'ki', 'de', 'da', 'çok', 'ama', 'fakat', 'lakin', 'ancak',
            'gerçekten', 'kesinlikle', 'tabii', 'tabi', 'şimdi', 'sonra', 'önce'
        ]

    def score_text(self, text):
        """
        Metin için kalite skoru hesaplar.

        Args:
            text: Değerlendirilecek metin

        Returns:
            float: 0 ile 1 arasında kalite skoru (1 = yüksek kalite)
        """
        if not text or len(text.strip()) == 0:
            return 0.0, {}

        # Metni temizle
        text = text.strip()

        # Çeşitli metin özelliklerini değerlendirelim
        features = {}

        # 1. Uzunluk puanı - Çok kısa veya çok uzun metinler düşük puan alır
        length = len(text.split())
        if length < 3:
            features['length_score'] = 0.1
        elif length < 5:
            features['length_score'] = 0.2
        elif length < 10:
            features['length_score'] = 0.4
        elif length < 20:
            features['length_score'] = 0.6
        elif length < 100:
            features['length_score'] = 0.8
        elif length < 500:
            features['length_score'] = 1.0
        elif length < 1000:
            features['length_score'] = 0.8
        else:
            features['length_score'] = 0.6

        # 2. Gramer ve yazım denetimi (Türkçe için uyarlanmış)
        # Türkçede noktalama işaretleri ve büyük harf kullanımı
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            features['grammar_score'] = 0.0
        else:
            # Cümlelerin büyük harfle başlayıp başlamadığını kontrol et
            correct_caps = sum(1 for s in sentences if s and s[0].isupper())
            caps_ratio = correct_caps / len(sentences) if sentences else 0

            # Noktalama işaretlerinin varlığını kontrol et
            punct_count = len(re.findall(r'[.!?,;:]', text))
            expected_punct = max(1, len(sentences) - 1)  # Beklenen minimum noktalama
            punct_ratio = min(1.0, punct_count / expected_punct) if expected_punct > 0 else 0

            # Türkçe'ye özgü yaygın yazım hatalarını kontrol et
            common_errors = [
                ('de da', 'de/da ayrı yazılmalı'),
                ('ki', 'ki bağlacı ayrı yazılmalı'),
                ('misin', 'soru eki ayrı yazılmalı'),
                ('geldimi', 'soru eki ayrı yazılmalı'),
                ('bişey', 'bir şey ayrı yazılmalı'),
                ('herşey', 'her şey ayrı yazılmalı'),
                ('hiçbirşey', 'hiçbir şey ayrı yazılmalı')
            ]

            error_count = sum(1 for error, _ in common_errors if error in text.lower())
            error_ratio = 1.0 - min(1.0, error_count / (len(text.split()) / 10 + 1))

            # Gramer puanını hesapla
            features['grammar_score'] = (caps_ratio * 0.4 + punct_ratio * 0.3 + error_ratio * 0.3)

        # 3. Kelime çeşitliliği (Türkçe için uyarlanmış)
        words = re.findall(r'\b\w+\b', text.lower())
        unique_words = set(words)
        if not words:
            features['diversity_score'] = 0.0
        else:
            # Türkçe metinler için çeşitlilik oranını ayarla
            diversity_ratio = len(unique_words) / len(words)
            # Türkçe'nin çekimli yapısı nedeniyle daha yüksek bir baz çeşitlilik beklenir
            features['diversity_score'] = min(1.0, diversity_ratio * 1.2)

        # 4. Özetlenebilirlik puanı - eğer pipeline varsa
        if self.pipeline and len(text.split()) > 20:
            try:
                # Pipeline özetleme mi yoksa çeviri mi ona göre işle
                if hasattr(self.pipeline, 'task') and self.pipeline.task == 'translation':
                    # Çeviri pipeline'ı, bu durumda çevirinin kalitesine bakma
                    translated = self.pipeline(text, max_length=100)[0]['translation_text']
                    features['summary_score'] = 0.7  # Varsayılan olarak iyi bir puan
                else:
                    # Özetleme pipeline'ı
                    max_length = min(128, max(30, len(text.split()) // 4))
                    summary = self.pipeline(text, max_length=max_length, min_length=10, do_sample=False)[0][
                        'generated_text']

                    # Özet ve orijinal metin arasındaki benzerliğe bakarak puan ver
                    summary_len = len(summary.split())
                    orig_len = len(text.split())

                    compression_ratio = summary_len / orig_len if orig_len > 0 else 0
                    if compression_ratio > 0.8 or compression_ratio < 0.05:
                        features['summary_score'] = 0.3
                    elif compression_ratio > 0.6 or compression_ratio < 0.1:
                        features['summary_score'] = 0.6
                    else:
                        features['summary_score'] = 0.9
            except Exception as e:
                logging.warning(f"Error during summarization: {str(e)}")
                features['summary_score'] = 0.5
        else:
            features['summary_score'] = 0.5

        # 5. Türkçe doldurma kelimeleri (filler words)
        filler_count = sum(1 for word in words if word.lower() in self.turkish_filler_words)
        if not words:
            features['filler_score'] = 1.0
        else:
            filler_ratio = filler_count / len(words)
            # Türkçede bazı doldurma kelimeleri doğal olabilir, bu yüzden daha toleranslı davran
            features['filler_score'] = 1.0 - min(1.0, filler_ratio * 3)

        # Puanları birleştir - farklı puanları ağırlıklandırarak
        weights = {
            'length_score': 0.15,
            'grammar_score': 0.3,  # Türkçe için gramere daha fazla ağırlık
            'diversity_score': 0.25,
            'summary_score': 0.2,
            'filler_score': 0.1
        }

        final_score = sum(features[key] * weights[key] for key in weights.keys())

        # Puanı 0-1 aralığına normalize et
        final_score = max(0.0, min(1.0, final_score))

        return final_score, features

    def batch_score(self, texts, batch_size=8):
        """
        Bir metin listesi için toplu kalite skoru hesaplar.

        Args:
            texts: Değerlendirilecek metin listesi
            batch_size: İşlenecek grup boyutu

        Returns:
            list: Kalite skorları listesi
        """
        results = []
        feature_results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            batch_features = []

            for text in batch_texts:
                score, features = self.score_text(text)
                batch_results.append(score)
                batch_features.append(features)

            results.extend(batch_results)
            feature_results.extend(batch_features)

        return results, feature_results
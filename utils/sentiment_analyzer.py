import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Türkçe metinler için duygu analizi yapan sınıf.
    Pozitif, negatif ve nötr duygu skorları üreterek metnin duygusal tonunu analiz eder.
    """

    def __init__(self, model=None, tokenizer=None):
        """
        Duygu analizi modülünü başlatır.

        Args:
            model: Duygu analizi modeli (isteğe bağlı)
            tokenizer: Model için tokenizer (isteğe bağlı)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_info = {"name": "Bilinmeyen Model", "language": "unknown"}

        if model is None or tokenizer is None:
            logger.info("Duygu analizi için varsayılan model yükleniyor...")
            self.load_default_model()

    def load_default_model(self):
        """Varsayılan duygu analizi modelini yükler"""
        try:
            # Türkçe duygu analizi modeli
            model_name = "savasy/bert-base-turkish-sentiment"
            logger.info(f"Türkçe duygu analizi modeli yükleniyor: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)

            self.model_info = {
                "name": model_name,
                "description": "Türkçe duygu analizi modeli",
                "language": "tr"
            }

            logger.info("Duygu analizi modeli başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Duygu analizi modeli yüklenemedi: {str(e)}")

            # Yedek model dene
            try:
                backup_model = "dbmdz/bert-base-turkish-cased"
                logger.info(f"Yedek Türkçe model deneniyor: {backup_model}")

                self.tokenizer = AutoTokenizer.from_pretrained(backup_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(backup_model)
                self.model.to(self.device)

                self.model_info = {
                    "name": backup_model,
                    "description": "Genel amaçlı Türkçe BERT modeli",
                    "language": "tr"
                }

                logger.info("Yedek model başarıyla yüklendi")
                return True
            except Exception as e2:
                logger.error(f"Yedek model yüklenemedi: {str(e2)}")
                raise e2

    def analyze_sentiment(self, text):
        """
        Metindeki duygu tonunu analiz eder.

        Args:
            text: Analiz edilecek metin

        Returns:
            dict: {
                'positive': float,  # Pozitif duygu skoru (0-1)
                'neutral': float,   # Nötr duygu skoru (0-1)
                'negative': float,  # Negatif duygu skoru (0-1)
                'dominant': str     # Baskın duygu (positive, neutral, negative)
                'score': float      # -1 (çok negatif) ile 1 (çok pozitif) arasında genel skor
            }
        """
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'dominant': 'neutral',
                'score': 0.0
            }

        try:
            # Metni tokenize et
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Sonuçları işle
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Model çıktısının formatına göre işlem yap
            if len(probabilities) >= 3:
                # 3 sınıflı model (negatif, nötr, pozitif)
                result = {
                    'negative': float(probabilities[0]),
                    'neutral': float(probabilities[1]),
                    'positive': float(probabilities[2])
                }
            else:
                # 2 sınıflı model (negatif, pozitif)
                result = {
                    'negative': float(probabilities[0]),
                    'neutral': 0.0,
                    'positive': float(probabilities[1])
                }

            # Baskın duyguyu belirle
            dominant_sentiment = max(result, key=result.get)
            result['dominant'] = dominant_sentiment

            # -1 ile 1 arasında genel bir skor hesapla
            # -1: çok negatif, 0: nötr, 1: çok pozitif
            weighted_score = result['positive'] - result['negative']
            result['score'] = float(weighted_score)

            return result

        except Exception as e:
            logger.error(f"Duygu analizi sırasında hata: {str(e)}")
            # Hata durumunda nötr sonuç döndür
            return {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'dominant': 'neutral',
                'score': 0.0
            }

    def batch_analyze(self, texts, batch_size=8):
        """
        Bir metin listesi için toplu duygu analizi yapar.

        Args:
            texts: Analiz edilecek metin listesi
            batch_size: İşlenecek grup boyutu

        Returns:
            list: Her metin için duygu analizi sonuçları
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.analyze_sentiment(text) for text in batch_texts]
            results.extend(batch_results)

        return results
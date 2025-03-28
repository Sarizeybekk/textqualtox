import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ToxicityScorer:
    def __init__(self, model=None, tokenizer=None):
        """
        Toxicity Scorer sınıfını başlatır.

        Args:
            model: Zararlılık modeli
            tokenizer: Model için tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_turkish_model = False

        if model is None or tokenizer is None:
            logging.warning("No toxicity model provided. Using default model.")
            self.load_default_model()

    def load_default_model(self):
        """
        Varsayılan zararlılık modelini yükler
        """
        try:
            # Öncelikle Türkçe duygu analizi modeli deneyelim
            model_name = "savasy/bert-base-turkish-sentiment"
            logging.info(f"Loading Turkish sentiment model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.is_turkish_model = True
            logging.info("Turkish sentiment model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading Turkish model: {str(e)}")
            try:
                # Yedek olarak genel model yükleyelim
                backup_model = "dbmdz/bert-base-turkish-cased"
                logging.info(f"Trying Turkish BERT model: {backup_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(backup_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(backup_model)
                self.model.to(self.device)
                self.is_turkish_model = True
                logging.info("Turkish BERT model loaded successfully")
            except Exception as e2:
                logging.error(f"Error loading Turkish BERT model: {str(e2)}")
                try:
                    # Son çare olarak İngilizce model kullanalım
                    english_model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                    logging.info(f"Trying English sentiment model: {english_model}")
                    self.tokenizer = AutoTokenizer.from_pretrained(english_model)
                    self.model = AutoModelForSequenceClassification.from_pretrained(english_model)
                    self.model.to(self.device)
                    self.is_turkish_model = False
                    logging.info("English sentiment model loaded successfully")
                except Exception as e3:
                    logging.error(f"Error loading English model: {str(e3)}")
                    raise e3

    def _contains_turkish_profanity(self, text):
        """
        Temel Türkçe küfür ve hakaret kontrolü yapar
        """
        # Türkçede yaygın küfür/hakaret içeren kelimelerin listesi
        turkish_profanity = [
            'aptal', 'salak', 'gerizekalı', 'ahmak', 'enayi', 'mal', 'geri zekalı',
            'beyinsiz', 'budala', 'adi', 'ahlaksız', 'şerefsiz', 'haysiyetsiz',
            'orospu', 'piç', 'yavşak', 'sürtük', 'sürtüğü', 'gavat', 'şerefsiz',
            'siktir', 'pezevenk', 'namussuz'
        ]

        # Noktalama işaretlerini ve sayıları kaldır
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\d+', '', text)
        words = text.split()

        # Metinde küfür/hakaret var mı kontrol et
        for word in turkish_profanity:
            if word in words:
                return True

        return False

    def _contains_negative_words(self, text):
        """
        Temel Türkçe olumsuz kelime kontrolü yapar
        """
        # Türkçede yaygın olumsuz kelimeler
        negative_words = [
            'kötü', 'berbat', 'rezalet', 'korkunç', 'iğrenç', 'üzücü', 'acı',
            'başarısız', 'yetersiz', 'düşük', 'zayıf', 'korkutucu', 'tehlikeli',
            'nefret', 'öfke', 'saldırgan', 'yanlış', 'hata', 'hayal kırıklığı'
        ]

        text = text.lower()
        count = sum(1 for word in negative_words if word in text.split())

        # Olumsuz kelime yoğunluğunu hesapla
        return count / len(text.split()) if text.split() else 0

    def score_text(self, text):
        """
        Metin için zararlılık skoru hesaplar.

        Args:
            text: Değerlendirilecek metin

        Returns:
            float: 0 ile 1 arasında zararlılık skoru (1 = çok zararlı)
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        # Temel kural tabanlı kontroller
        profanity_detected = self._contains_turkish_profanity(text)
        negative_ratio = self._contains_negative_words(text)

        if profanity_detected:
            base_score = 0.8  # Küfür/hakaret varsa yüksek başlangıç skoru
        else:
            base_score = negative_ratio * 0.5  # Olumsuz kelime yoğunluğuna göre skor

        try:
            # Model tabanlı skorlama
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Modele göre doğru şekilde skoru alalım
            if self.is_turkish_model:
                # Türkçe duygu analizi modeli için özel işlem
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

                # savasy/bert-base-turkish-sentiment için:
                # 0: negative, 1: neutral, 2: positive
                if len(probs) >= 3:
                    # Negatif olasılığını zararlılık skoru olarak kullan ama çok yüksek değerler üretmemesi için 0.7 ile çarp
                    model_score = probs[0] * 0.7
                else:
                    # İki sınıflı model için
                    model_score = probs[0] * 0.6
            else:
                # İngilizce model için
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                # İngilizce modeller genellikle Türkçe için çok yüksek sonuçlar verir, bu yüzden 0.5 ile çarp
                model_score = probs[0] * 0.5

            # Kural tabanlı skor ve model skor birleşimi
            final_score = (base_score * 0.4) + (model_score * 0.6)

            # 0-1 aralığına sınırla
            final_score = max(0.0, min(1.0, final_score))

            return final_score

        except Exception as e:
            logging.error(f"Error scoring toxicity: {str(e)}")
            # Hata durumunda sadece kural tabanlı skoru döndür
            return min(base_score, 1.0)

    def batch_score(self, texts, batch_size=16):
        """
        Bir metin listesi için toplu zararlılık skoru hesaplar.

        Args:
            texts: Değerlendirilecek metin listesi
            batch_size: İşlenecek grup boyutu

        Returns:
            list: Zararlılık skorları listesi
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_scores = [self.score_text(text) for text in batch_texts]
            results.extend(batch_scores)

        return results
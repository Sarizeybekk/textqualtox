import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import json
from typing import List, Dict, Any, Tuple, Optional, Union
import threading

# Loglama yapılandırması
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelSelector:
    """
    Farklı NLP modellerini değerlendirip en iyi performansı sağlayanı seçen sınıf.
    """

    def __init__(self, cache_dir: Optional[str] = None, use_cache: bool = True) -> None:
        """
        Model Seçici sınıfını başlatır.

        Args:
            cache_dir: Model ve değerlendirme sonuçlarının önbelleğe alınacağı dizin
            use_cache: Önbellek kullanımını etkinleştir/devre dışı bırak
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.toxicity_models = []
        self.quality_models = []
        self.best_toxicity_model = None
        self.best_quality_model = None
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".text_quality_toxicity_cache")

        # Önbellek dizinini oluştur
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(
            f"ModelSelector başlatıldı. Cihaz: {self.device}, Önbellek: {'Etkin' if use_cache else 'Devre dışı'}")

    def load_candidate_models(self) -> bool:
        """
        Değerlendirme için aday modelleri yükler.

        Returns:
            bool: Yükleme başarılı mı?
        """
        # Önbellekten okunan modeller
        cached_models = self._load_from_cache()
        if cached_models:
            self.toxicity_models = cached_models.get("toxicity_models", [])
            self.quality_models = cached_models.get("quality_models", [])
            self.best_toxicity_model = cached_models.get("best_toxicity_model")
            self.best_quality_model = cached_models.get("best_quality_model")

            # Önbellek bulundu, doğrulama yap
            if self.best_toxicity_model and self.best_quality_model:
                logger.info("Önbellekten en iyi modeller yüklendi.")
                # Tokenizer ve model erişilebilir mi kontrol et
                if "tokenizer" in self.best_toxicity_model and "model" in self.best_toxicity_model:
                    # En iyi model önbellekten doğru yüklendi
                    return True

        # Zararlılık modelleri - Türkçe için optimize edilmiş
        toxicity_candidates = [
            {"name": "savasy/bert-base-turkish-sentiment", "type": "sentiment",
             "language": "tr", "priority": 1, "description": "Türkçe duygu analizi için BERT modeli"},
            {"name": "loodos/electra-turkish-sentiment", "type": "sentiment",
             "language": "tr", "priority": 2, "description": "Türkçe duygu analizi için ELECTRA modeli"},
            {"name": "dbmdz/bert-base-turkish-cased", "type": "general",
             "language": "tr", "priority": 3, "description": "Genel amaçlı Türkçe BERT modeli"},
            {"name": "ytu-ce-cosmos/turkish-bert-uncased-toxicity", "type": "toxicity",
             "language": "tr", "priority": 4, "description": "Türkçe zararlılık tespiti için BERT modeli",
             "optional": True},
            {"name": "unitary/toxic-bert", "type": "toxicity",
             "language": "en", "priority": 5, "description": "İngilizce zararlılık tespiti BERT modeli"}
        ]

        # Kalite modelleri
        quality_candidates = [
            {"name": "Helsinki-NLP/opus-mt-tr-en", "type": "translation",
             "language": "tr", "priority": 1, "description": "Türkçe-İngilizce çeviri modeli"},
            {"name": "tuner/pegasus-turkish", "type": "summarization",
             "language": "tr", "priority": 2, "description": "Türkçe özetleme için PEGASUS modeli", "optional": True},
            {"name": "dbmdz/t5-base-turkish-summarization", "type": "summarization",
             "language": "tr", "priority": 3, "description": "Türkçe özetleme için T5 modeli", "optional": True},
            {"name": "sshleifer/distilbart-cnn-6-6", "type": "summarization",
             "language": "en", "priority": 4, "description": "İngilizce özetleme için DistilBART modeli"}
        ]

        # Modelleri önceliğe göre sırala
        toxicity_candidates.sort(key=lambda x: x.get("priority", 999))
        quality_candidates.sort(key=lambda x: x.get("priority", 999))

        # Zaman aşımını önlemek için asenkron model yükleme
        self._load_models_async(toxicity_candidates, quality_candidates)

        # Yeterli model yüklendi mi kontrol et
        return len(self.toxicity_models) > 0 and len(self.quality_models) > 0

    def _load_models_async(self, toxicity_candidates: List[Dict[str, Any]],
                           quality_candidates: List[Dict[str, Any]]) -> None:
        """
        Modelleri asenkron olarak yükler

        Args:
            toxicity_candidates: Zararlılık modellerinin listesi
            quality_candidates: Kalite modellerinin listesi
        """
        # Zararlılık modelleri için eş zamanlı yükleme
        toxicity_threads = []
        for candidate in toxicity_candidates:
            thread = threading.Thread(
                target=self._load_toxicity_model,
                args=(candidate,)
            )
            thread.start()
            toxicity_threads.append(thread)

        # Kalite modelleri için eş zamanlı yükleme
        quality_threads = []
        for candidate in quality_candidates:
            thread = threading.Thread(
                target=self._load_quality_model,
                args=(candidate,)
            )
            thread.start()
            quality_threads.append(thread)

        # Tüm işlemlerin tamamlanmasını bekle (timeout ile)
        for thread in toxicity_threads:
            thread.join(timeout=60)  # Her model için 60 saniye timeout

        for thread in quality_threads:
            thread.join(timeout=60)  # Her model için 60 saniye timeout

    def _load_toxicity_model(self, candidate: Dict[str, Any]) -> None:
        """
        Bir zararlılık modelini yükler

        Args:
            candidate: Model bilgilerini içeren sözlük
        """
        try:
            logger.info(f"Zararlılık modeli yükleniyor: {candidate['name']}")
            start_time = time.time()

            tokenizer = AutoTokenizer.from_pretrained(candidate["name"])
            model = AutoModelForSequenceClassification.from_pretrained(candidate["name"])
            model.to(self.device)

            load_time = time.time() - start_time
            candidate["tokenizer"] = tokenizer
            candidate["model"] = model
            candidate["load_time"] = load_time

            # Liste eşzamanlı erişim için koruma
            with threading.Lock():
                self.toxicity_models.append(candidate)

            logger.info(f"{candidate['name']} modeli {load_time:.2f} saniyede başarıyla yüklendi")
        except Exception as e:
            if candidate.get("optional", False):
                logger.warning(f"Opsiyonel model {candidate['name']} atlanıyor: {str(e)}")
            else:
                logger.error(f"{candidate['name']} yüklenirken hata: {str(e)}")

    def _load_quality_model(self, candidate: Dict[str, Any]) -> None:
        """
        Bir kalite modelini yükler

        Args:
            candidate: Model bilgilerini içeren sözlük
        """
        try:
            logger.info(f"Kalite modeli yükleniyor: {candidate['name']}")
            start_time = time.time()

            if candidate["type"] == "translation":
                pipe = pipeline("translation", model=candidate["name"], device=0 if self.device == "cuda" else -1)
            elif candidate["type"] == "summarization":
                pipe = pipeline("summarization", model=candidate["name"], device=0 if self.device == "cuda" else -1)

            load_time = time.time() - start_time
            candidate["pipeline"] = pipe
            candidate["load_time"] = load_time

            # Liste eşzamanlı erişim için koruma
            with threading.Lock():
                self.quality_models.append(candidate)

            logger.info(f"{candidate['name']} modeli {load_time:.2f} saniyede başarıyla yüklendi")
        except Exception as e:
            if candidate.get("optional", False):
                logger.warning(f"Opsiyonel model {candidate['name']} atlanıyor: {str(e)}")
            else:
                logger.error(f"{candidate['name']} yüklenirken hata: {str(e)}")

    def evaluate_toxicity_models(self, validation_texts: List[str],
                                 validation_labels: List[int]) -> Optional[Dict[str, Any]]:
        """
        Zararlılık modellerini değerlendirir ve en iyisini seçer

        Args:
            validation_texts: Doğrulama metinleri
            validation_labels: Doğrulama etiketleri (1=zararlı, 0=zararsız)

        Returns:
            Dict[str, Any]: En iyi modelin bilgileri veya None
        """
        if not self.toxicity_models:
            logger.error("Değerlendirme için zararlılık modeli yüklenmemiş")
            return None

        results = []

        for model_info in self.toxicity_models:
            logger.info(f"Zararlılık modeli değerlendiriliyor: {model_info['name']}")
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]

            predictions = []
            start_time = time.time()

            try:
                for text in validation_texts:
                    # Metni tokenize et
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    inputs = {key: val.to(self.device) for key, val in inputs.items()}

                    # Tahmin yap
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Sonucu almak için model tipine göre işlem yap
                    if model_info["type"] == "sentiment":
                        # Sentiment modellerinde genellikle 0=negatif, 1=nötr, 2=pozitif
                        # veya 0=negatif, 1=pozitif
                        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                        if len(probs) >= 3:
                            # Negatif olasılığını zararlılık olarak kabul et
                            pred = 1 if probs[0] > 0.5 else 0
                        else:
                            # İki sınıflı model
                            pred = 1 if probs[0] > 0.5 else 0
                    elif model_info["type"] == "toxicity":
                        # Toxicity modelleri genellikle 0=non-toxic, 1=toxic
                        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                        pred = 1 if probs[1] > 0.5 else 0
                    else:
                        # Genel model - varsayılan
                        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
                        pred = 1 if probs[0] > 0.5 else 0

                    predictions.append(pred)

                eval_time = time.time() - start_time

                # Performans metrikleri hesapla
                accuracy = accuracy_score(validation_labels, predictions)
                precision = precision_score(validation_labels, predictions, average='binary', zero_division=0)
                recall = recall_score(validation_labels, predictions, average='binary', zero_division=0)
                f1 = f1_score(validation_labels, predictions, average='binary', zero_division=0)

                # F1, precision ve recall'un ağırlıklı ortalaması
                weighted_score = (f1 * 0.5) + (precision * 0.3) + (recall * 0.2)

                # Sonuçları kaydet
                evaluation_result = {
                    "model": model_info,
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "weighted_score": float(weighted_score),
                    "eval_time": float(eval_time),
                    "predictions": predictions
                }

                results.append(evaluation_result)

                logger.info(
                    f"{model_info['name']} - Doğruluk: {accuracy:.4f}, "
                    f"F1: {f1:.4f}, Precision: {precision:.4f}, "
                    f"Recall: {recall:.4f}, Süre: {eval_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"{model_info['name']} değerlendirilirken hata: {str(e)}")

        if not results:
            logger.error("Hiçbir model değerlendirilemedi")
            return None

        # Sıralama ve en iyi modeli seç (ağırlıklı skora göre)
        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        best_model = results[0]["model"]

        logger.info(
            f"En iyi zararlılık modeli: {best_model['name']} - "
            f"Ağırlıklı skor: {results[0]['weighted_score']:.4f}, "
            f"F1: {results[0]['f1_score']:.4f}"
        )

        self.best_toxicity_model = best_model

        # Önbelleğe kaydet
        if self.use_cache:
            self._save_to_cache()

        return best_model

    def evaluate_quality_models(self, validation_texts: List[str],
                                reference_summaries: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Kalite modellerini değerlendirir ve en iyisini seçer

        Args:
            validation_texts: Doğrulama metinleri
            reference_summaries: Referans özetler (opsiyonel)

        Returns:
            Dict[str, Any]: En iyi modelin bilgileri veya None
        """
        if not self.quality_models:
            logger.error("Değerlendirme için kalite modeli yüklenmemiş")
            return None

        results = []

        for model_info in self.quality_models:
            logger.info(f"Kalite modeli değerlendiriliyor: {model_info['name']}")
            pipe = model_info["pipeline"]

            start_time = time.time()
            processing_success = 0
            avg_processing_time = []

            # Her metni değerlendir
            for i, text in enumerate(validation_texts[:5]):  # Performans için sadece ilk 5 metni değerlendir
                try:
                    text_start_time = time.time()

                    if model_info["type"] == "translation":
                        _ = pipe(text, max_length=100)
                    elif model_info["type"] == "summarization":
                        # Çok kısa metinlerde sorun oluşmaması için metin uzunluğunu kontrol et
                        text_words = len(text.split())
                        max_length = min(100, max(30, text_words // 2))
                        min_length = min(30, max(5, text_words // 4))
                        _ = pipe(text, max_length=max_length, min_length=min_length, do_sample=False)

                    text_process_time = time.time() - text_start_time
                    avg_processing_time.append(text_process_time)
                    processing_success += 1

                except Exception as e:
                    logger.warning(f"{model_info['name']} için metin {i} işlenirken hata: {str(e)}")

            eval_time = time.time() - start_time
            success_rate = processing_success / min(5, len(validation_texts))
            avg_time = np.mean(avg_processing_time) if avg_processing_time else float('inf')

            # Modelin tipine göre skoru ayarla
            if model_info["language"] == "tr":
                # Türkçe modeller için daha yüksek ağırlık
                language_weight = 1.2
            else:
                language_weight = 0.8

            # Sonuçları kaydet
            evaluation_result = {
                "model": model_info,
                "success_rate": float(success_rate),
                "avg_processing_time": float(avg_time),
                "eval_time": float(eval_time),
                "language_weight": float(language_weight)
            }

            results.append(evaluation_result)

            logger.info(
                f"{model_info['name']} - Başarı Oranı: {success_rate:.2f}, "
                f"Ortalama İşleme Süresi: {avg_time:.4f}s, "
                f"Toplam Süre: {eval_time:.2f}s"
            )

        if not results:
            logger.error("Hiçbir kalite modeli değerlendirilemedi")
            return None

        # Sıralama ve en iyi modeli seç
        # Başarı oranı, dil ağırlığı ve hız faktörlerini dengeleyen bir formül
        for result in results:
            result["score"] = (
                    result["success_rate"] * 0.6 +
                    (1 / (1 + result["avg_processing_time"])) * 0.2 +
                    result["language_weight"] * 0.2
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        best_model = results[0]["model"]

        logger.info(
            f"En iyi kalite modeli: {best_model['name']} - "
            f"Skor: {results[0]['score']:.4f}, "
            f"Başarı Oranı: {results[0]['success_rate']:.2f}"
        )

        self.best_quality_model = best_model

        # Önbelleğe kaydet
        if self.use_cache:
            self._save_to_cache()

        return best_model

    def select_best_models(self, sample_texts: List[str]) -> bool:
        """
        En iyi modelleri örnek veriyle otomatik olarak seçer

        Args:
            sample_texts: Örnek metinler listesi

        Returns:
            bool: Model seçimi başarılı mı?
        """
        # Modeller önbellekten yüklendiyse doğrudan kullan
        if self.best_toxicity_model and self.best_quality_model:
            logger.info("En iyi modeller önbellekten kullanılıyor")
            return True

        if not self.toxicity_models or not self.quality_models:
            if not self.load_candidate_models():
                logger.error("Aday modeller yüklenemedi")
                return False

        # Basit zararlılık değerlendirmesi için örnekler
        toxic_samples = [
            "Bu insanlar gerçekten çok aptal! Hepsi geri zekalı ve cahil.",
            "Senden nefret ediyorum, kaybol buradan!",
            "sen ne anlarsın ki bu konudan? gerizekalılar bile senden daha mantıklı konuşur"
        ]

        non_toxic_samples = [
            "Türkiye, zengin tarihi ve kültürel mirası ile güzel bir ülkedir.",
            "Bugün hava çok güzel. Parkta yürüyüş yaptım ve kuşları izledim.",
            "Bilgisayarınızı hızlandırmak için gereksiz programları kaldırın."
        ]

        # Doğrulama verisi hazırla
        validation_texts = toxic_samples + non_toxic_samples
        if sample_texts and len(sample_texts) > 0:
            # Kullanıcı örneklerinden bir kısmını ekle (en fazla 5 tane)
            validation_texts.extend(sample_texts[:5])

        validation_labels = [1, 1, 1, 0, 0, 0] + [0] * min(5, len(sample_texts))

        # En iyi zararlılık modelini seç
        best_toxicity = self.evaluate_toxicity_models(validation_texts, validation_labels)

        # En iyi kalite modelini seç
        best_quality = self.evaluate_quality_models(validation_texts)

        success = best_toxicity is not None and best_quality is not None

        if success and self.use_cache:
            self._save_to_cache()

        return success

    def get_best_models(self) -> Dict[str, Any]:
        """
        Seçilen en iyi modelleri döndürür

        Returns:
            Dict[str, Any]: En iyi modellerin bilgileri
        """
        return {
            "toxicity_model": self.best_toxicity_model["model"] if self.best_toxicity_model else None,
            "toxicity_tokenizer": self.best_toxicity_model["tokenizer"] if self.best_toxicity_model else None,
            "quality_pipeline": self.best_quality_model["pipeline"] if self.best_quality_model else None,
            "toxicity_model_info": {
                "name": self.best_toxicity_model["name"] if self.best_toxicity_model else "Unknown",
                "description": self.best_toxicity_model["description"] if self.best_toxicity_model else "Unknown",
                "language": self.best_toxicity_model["language"] if self.best_toxicity_model else "Unknown"
            },
            "quality_model_info": {
                "name": self.best_quality_model["name"] if self.best_quality_model else "Unknown",
                "description": self.best_quality_model["description"] if self.best_quality_model else "Unknown",
                "language": self.best_quality_model["language"] if self.best_quality_model else "Unknown"
            }
        }

    def _save_to_cache(self) -> None:
        """Modellerin değerlendirme sonuçlarını önbelleğe kaydeder"""
        if not self.use_cache:
            return

        try:
            cache_file = os.path.join(self.cache_dir, "model_selection_results.json")

            # Modeller ve tokenizer'ları hariç tutarak kalan bilgileri kaydet
            cache_data = {
                "timestamp": time.time(),
                "toxicity_models": [
                    {k: v for k, v in model.items() if k not in ["model", "tokenizer"]}
                    for model in self.toxicity_models
                ],
                "quality_models": [
                    {k: v for k, v in model.items() if k not in ["pipeline"]}
                    for model in self.quality_models
                ],
                "best_toxicity_model": (
                    {k: v for k, v in self.best_toxicity_model.items() if k not in ["model", "tokenizer"]}
                    if self.best_toxicity_model else None
                ),
                "best_quality_model": (
                    {k: v for k, v in self.best_quality_model.items() if k not in ["pipeline"]}
                    if self.best_quality_model else None
                )
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Model seçim sonuçları önbelleğe kaydedildi: {cache_file}")
        except Exception as e:
            logger.error(f"Önbelleğe kaydetme hatası: {str(e)}")

    def _load_from_cache(self) -> Optional[Dict[str, Any]]:
        """
        Önbellekten modellerin değerlendirme sonuçlarını yükler

        Returns:
            Optional[Dict[str, Any]]: Yüklenen önbellek verisi veya None
        """
        if not self.use_cache:
            return None

        try:
            cache_file = os.path.join(self.cache_dir, "model_selection_results.json")

            if not os.path.exists(cache_file):
                return None

            # Önbellek dosyasının yaşını kontrol et (24 saatten eskiyse yok say)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > 86400:  # 24 saat = 86400 saniye
                logger.info(f"Önbellek dosyası çok eski ({file_age / 3600:.1f} saat), tekrar değerlendirme yapılacak")
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            logger.info(f"Model seçim sonuçları önbellekten yüklendi: {cache_file}")

            # Önbellekten sadece isim bilgilerini yükle, modellerin kendisini değil
            return cache_data

        except Exception as e:
            logger.error(f"Önbellekten yükleme hatası: {str(e)}")
            return None
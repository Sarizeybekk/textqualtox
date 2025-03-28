import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging
import os
import json
import time
from typing import List, Dict, Any, Tuple, Optional, Union
from models.model_selector import ModelSelector

# Loglama yapılandırması
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ModelManager:
    """
    Metin kalitesi ve zararlılık değerlendirmesi için NLP modellerini yöneten sınıf.
    Bu sınıf, modellerin yüklenmesi, değerlendirilmesi ve seçilmesinden sorumludur.
    """

    def __init__(self, cache_dir: Optional[str] = None, use_cache: bool = True) -> None:
        """
        Model Yöneticisi sınıfını başlatır.

        Args:
            cache_dir: Modellerin önbelleğe alınacağı dizin
            use_cache: Önbellek kullanımını etkinleştir/devre dışı bırak
        """
        self.toxicity_model = None
        self.toxicity_tokenizer = None
        self.quality_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".text_quality_toxicity_cache")
        self.use_cache = use_cache
        self.model_selector = ModelSelector(cache_dir=self.cache_dir, use_cache=self.use_cache)
        self.model_info = {
            "toxicity": {
                "name": "Bilinmeyen Model",
                "description": "Model henüz yüklenmedi",
                "language": "unknown"
            },
            "quality": {
                "name": "Bilinmeyen Model",
                "description": "Model henüz yüklenmedi",
                "language": "unknown"
            }
        }

        # Önbellek dizinini oluştur
        if self.use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(
            f"ModelManager başlatıldı. Cihaz: {self.device}, Önbellek: {'Etkin' if use_cache else 'Devre dışı'}")

    def load_models_auto_select(self, sample_texts: Optional[List[str]] = None) -> bool:
        """
        En iyi modelleri otomatik olarak seçerek yükler.

        Args:
            sample_texts: Model seçimi için örnek metinler

        Returns:
            bool: Yükleme başarılı mı?
        """
        if sample_texts is None or len(sample_texts) == 0:
            # Örnek metinler yoksa varsayılan örnekler kullan
            sample_texts = [
                "Türkiye, zengin tarihi ve kültürel mirası ile güzel bir ülkedir.",
                "Bu ürün tam bir hayal kırıklığı! Paramı geri istiyorum!",
                "Bugün hava çok güzel. Parkta yürüyüş yaptım ve kuşları izledim.",
                "Sen ne anlarsın ki bu konudan? Boş konuşma artık!"
            ]

        try:
            logger.info("Otomatik model seçimi başlatılıyor...")
            start_time = time.time()

            success = self.model_selector.select_best_models(sample_texts)

            if success:
                best_models = self.model_selector.get_best_models()
                self.toxicity_model = best_models["toxicity_model"]
                self.toxicity_tokenizer = best_models["toxicity_tokenizer"]
                self.quality_pipeline = best_models["quality_pipeline"]
                self.model_info["toxicity"] = best_models["toxicity_model_info"]
                self.model_info["quality"] = best_models["quality_model_info"]

                selection_time = time.time() - start_time
                logger.info(f"Otomatik model seçimi {selection_time:.2f} saniyede tamamlandı")
                logger.info(f"Seçilen zararlılık modeli: {self.model_info['toxicity']['name']}")
                logger.info(f"Seçilen kalite modeli: {self.model_info['quality']['name']}")

                # Kullanılan modelleri önbelleğe kaydet
                if self.use_cache:
                    self._save_models_to_cache()

                return True
            else:
                logger.error("Otomatik model seçimi başarısız oldu, varsayılan modellere dönülüyor")
                return self.load_default_models()

        except Exception as e:
            logger.error(f"Otomatik model seçimi sırasında hata: {str(e)}")
            return self.load_default_models()

    def load_default_models(self) -> bool:
        """
        Varsayılan modelleri yükler (otomatik seçim başarısız olursa).

        Returns:
            bool: Yükleme başarılı mı?
        """
        logger.info("Varsayılan modeller yükleniyor...")

        # Önce önbellekten yüklemeyi dene
        if self.use_cache and self._load_models_from_cache():
            logger.info("Modeller önbellekten başarıyla yüklendi")
            return True

        success_toxicity = self.load_toxicity_model()
        success_quality = self.load_quality_model()

        overall_success = success_toxicity and success_quality

        if overall_success and self.use_cache:
            self._save_models_to_cache()

        return overall_success

    def load_toxicity_model(self, model_name: str = "savasy/bert-base-turkish-sentiment") -> bool:
        """
        Zararlılık tespiti için model yükleme.

        Args:
            model_name: Yüklenecek model ismi

        Returns:
            bool: Yükleme başarılı mı?
        """
        try:
            logger.info(f"Zararlılık modeli yükleniyor: {model_name}")
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model_info["toxicity"]["name"] = model_name
            self.model_info["toxicity"]["description"] = "Türkçe duygu analizi modeli"
            self.model_info["toxicity"]["language"] = "tr"
            logger.info("Zararlılık modeli başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Zararlılık modeli yüklenirken hata: {str(e)}")
            # Alternatif model deneyelim
            try:
                backup_model = "dbmdz/bert-base-turkish-cased"
                logger.info(f"Yedek Türkçe model deneniyor: {backup_model}")
                self.toxicity_tokenizer = AutoTokenizer.from_pretrained(backup_model)
                self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(backup_model)
                self.model_info["toxicity"]["name"] = backup_model
                self.model_info["toxicity"]["description"] = "Genel amaçlı Türkçe BERT modeli"
                self.model_info["toxicity"]["language"] = "tr"
                logger.info("Yedek Türkçe model başarıyla yüklendi")
                return True
            except Exception as e2:
                logger.error(f"Yedek Türkçe model yüklenirken hata: {str(e2)}")
                try:
                    fallback_model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                    logger.info(f"İngilizce duygu analizi modeli deneniyor: {fallback_model}")
                    self.toxicity_tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                    self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(fallback_model)
                    self.model_info["toxicity"]["name"] = fallback_model
                    self.model_info["toxicity"]["description"] = "İngilizce duygu analizi modeli"
                    self.model_info["toxicity"]["language"] = "en"
                    logger.info("İngilizce duygu analizi modeli başarıyla yüklendi")
                    return True
                except Exception as e3:
                    logger.error(f"İngilizce model yüklenirken hata: {str(e3)}")
                    return False

    def load_quality_model(self, model_name: str = "sshleifer/distilbart-cnn-6-6") -> bool:
        """
        Metin kalitesi değerlendirmesi için model yükleme.

        Args:
            model_name: Yüklenecek model ismi

        Returns:
            bool: Yükleme başarılı mı?
        """
        try:
            logger.info(f"Kalite modeli yükleniyor: {model_name}")
            self.quality_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1
            )
            self.model_info["quality"]["name"] = model_name
            self.model_info["quality"]["description"] = "İngilizce metin özetleme modeli"
            self.model_info["quality"]["language"] = "en"
            logger.info("Kalite modeli başarıyla yüklendi")
            return True
        except Exception as e:
            logger.error(f"Kalite modeli yüklenirken hata: {str(e)}")

            # Daha hafif bir model deneyelim
            try:
                backup_model = "Helsinki-NLP/opus-mt-tr-en"
                logger.info(f"Türkçe çeviri modeli deneniyor: {backup_model}")
                self.quality_pipeline = pipeline(
                    "translation",
                    model=backup_model,
                    device=0 if self.device == "cuda" else -1
                )
                self.model_info["quality"]["name"] = backup_model
                self.model_info["quality"]["description"] = "Türkçe-İngilizce çeviri modeli"
                self.model_info["quality"]["language"] = "tr"
                logger.info("Türkçe çeviri modeli başarıyla yüklendi")
                return True
            except Exception as e2:
                logger.error(f"Türkçe çeviri modeli yüklenirken hata: {str(e2)}")
                try:
                    light_model = "sshleifer/distilbart-xsum-12-6"
                    logger.info(f"Daha hafif özetleme modeli deneniyor: {light_model}")
                    self.quality_pipeline = pipeline(
                        "text2text-generation",
                        model=light_model,
                        device=0 if self.device == "cuda" else -1
                    )
                    self.model_info["quality"]["name"] = light_model
                    self.model_info["quality"]["description"] = "Hafif İngilizce özetleme modeli"
                    self.model_info["quality"]["language"] = "en"
                    logger.info("Hafif özetleme modeli başarıyla yüklendi")
                    return True
                except Exception as e3:
                    logger.error(f"Hafif özetleme modeli yüklenirken hata: {str(e3)}")
                    return False

    def _save_models_to_cache(self) -> None:
        """Kullanılan modellerin bilgilerini önbelleğe kaydeder."""
        if not self.use_cache:
            return

        try:
            cache_file = os.path.join(self.cache_dir, "model_manager_state.json")

            cache_data = {
                "timestamp": time.time(),
                "model_info": self.model_info
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Model bilgileri önbelleğe kaydedildi: {cache_file}")
        except Exception as e:
            logger.error(f"Önbelleğe kaydetme hatası: {str(e)}")

    def _load_models_from_cache(self) -> bool:
        """
        Önbellekten model bilgilerini yükler.

        Returns:
            bool: Yükleme başarılı mı?
        """
        if not self.use_cache:
            return False

        try:
            cache_file = os.path.join(self.cache_dir, "model_manager_state.json")

            if not os.path.exists(cache_file):
                return False

            # Önbellek dosyasının yaşını kontrol et (24 saatten eskiyse yok say)
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age > 86400:  # 24 saat = 86400 saniye
                logger.info(f"Önbellek dosyası çok eski ({file_age / 3600:.1f} saat), yeniden yükleme yapılacak")
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Model bilgilerini önbellekten al
            self.model_info = cache_data.get("model_info", self.model_info)

            # Önbellekteki bilgilerle yeniden yükleme yap
            toxicity_name = self.model_info["toxicity"].get("name")
            quality_name = self.model_info["quality"].get("name")

            toxicity_success = self.load_toxicity_model(toxicity_name) if toxicity_name else False
            quality_success = self.load_quality_model(quality_name) if quality_name else False

            return toxicity_success and quality_success

        except Exception as e:
            logger.error(f"Önbellekten model yükleme hatası: {str(e)}")
            return False

    def get_models(self) -> Dict[str, Any]:
        """
        Yüklenen modelleri ve bilgilerini döndürür.

        Returns:
            Dict[str, Any]: Model, tokenizer, pipeline ve model bilgileri
        """
        return {
            "toxicity_model": self.toxicity_model,
            "toxicity_tokenizer": self.toxicity_tokenizer,
            "quality_pipeline": self.quality_pipeline,
            "model_info": self.model_info
        }
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelManager:
    def __init__(self):
        self.toxicity_model = None
        self.toxicity_tokenizer = None
        self.quality_model = None
        self.quality_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")

    def load_toxicity_model(self, model_name="savasy/bert-base-turkish-sentiment"):
        """
        Zararlılık tespiti için model yükleme
        """
        try:
            logging.info(f"Loading toxicity model: {model_name}")
            self.toxicity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logging.info("Toxicity model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading toxicity model: {str(e)}")
            # Alternatif model deneyelim
            try:
                backup_model = "dbmdz/bert-base-turkish-cased"
                logging.info(f"Trying backup Turkish model: {backup_model}")
                self.toxicity_tokenizer = AutoTokenizer.from_pretrained(backup_model)
                self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(backup_model)
                logging.info("Backup Turkish model loaded successfully")
                return True
            except Exception as e2:
                logging.error(f"Error loading backup Turkish model: {str(e2)}")
                try:
                    english_model = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
                    logging.info(f"Trying English sentiment model: {english_model}")
                    self.toxicity_tokenizer = AutoTokenizer.from_pretrained(english_model)
                    self.toxicity_model = AutoModelForSequenceClassification.from_pretrained(english_model)
                    logging.info("English sentiment model loaded successfully")
                    return True
                except Exception as e3:
                    logging.error(f"Error loading English model: {str(e3)}")
                    return False

    def load_quality_model(self, model_name="sshleifer/distilbart-cnn-6-6"):
        """
        Metin kalitesi değerlendirmesi için model yükleme (özetleme kapasitesi olan bir model)
        """
        try:
            logging.info(f"Loading quality model: {model_name}")
            self.quality_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device == "cuda" else -1
            )
            logging.info("Quality model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading quality model: {str(e)}")

            # Daha hafif bir model deneyelim
            try:
                backup_model = "Helsinki-NLP/opus-mt-tc-big-tr-en"
                logging.info(f"Trying Turkish translation model for quality: {backup_model}")
                self.quality_pipeline = pipeline(
                    "translation",
                    model=backup_model,
                    tokenizer=backup_model,
                    device=0 if self.device == "cuda" else -1
                )
                logging.info("Turkish translation model loaded successfully")
                return True
            except Exception as e2:
                logging.error(f"Error loading Turkish translation model: {str(e2)}")
                try:
                    light_model = "sshleifer/distilbart-xsum-12-6"
                    logging.info(f"Trying lighter quality model: {light_model}")
                    self.quality_pipeline = pipeline(
                        "text2text-generation",
                        model=light_model,
                        tokenizer=light_model,
                        device=0 if self.device == "cuda" else -1
                    )
                    logging.info("Lighter quality model loaded successfully")
                    return True
                except Exception as e3:
                    logging.error(f"Error loading lighter quality model: {str(e3)}")
                    return False

    def get_models(self):
        """
        Yüklenen modelleri döndürür
        """
        return {
            "toxicity_model": self.toxicity_model,
            "toxicity_tokenizer": self.toxicity_tokenizer,
            "quality_pipeline": self.quality_pipeline if hasattr(self, 'quality_pipeline') else None
        }
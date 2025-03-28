import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataHandler:
    def __init__(self, quality_scorer=None, toxicity_scorer=None):
        """
        Veri işleme sınıfını başlatır.

        Args:
            quality_scorer: Metin kalitesi değerlendirme nesnesi
            toxicity_scorer: Zararlılık skoru değerlendirme nesnesi
        """
        self.quality_scorer = quality_scorer
        self.toxicity_scorer = toxicity_scorer

    def load_data(self, file_path, text_column=None):
        """
        CSV veya Excel dosyasını yükler.

        Args:
            file_path: Yüklenecek dosyanın yolu
            text_column: Metin sütunu adı (belirtilmezse otomatik tespit edilir)

        Returns:
            pd.DataFrame: Yüklenen veri
        """
        try:
            # Dosya uzantısını kontrol et
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("Desteklenmeyen dosya formatı. Lütfen CSV veya Excel dosyası yükleyin.")

            # Metin sütununu belirle
            if text_column is None:
                # En çok metin içeriği olan sütunu bul
                text_lengths = {}
                for col in df.columns:
                    if df[col].dtype == object:  # Sadece metin sütunlarını kontrol et
                        # Ortalama metin uzunluğunu hesapla
                        avg_len = df[col].astype(str).str.len().mean()
                        text_lengths[col] = avg_len

                if text_lengths:
                    # En uzun ortalama metne sahip sütunu seç
                    text_column = max(text_lengths.items(), key=lambda x: x[1])[0]
                else:
                    # Hiçbir metin sütunu bulunamazsa ilk sütunu kullan
                    text_column = df.columns[0]

                logging.info(f"Otomatik tespit edilen metin sütunu: {text_column}")

            return df, text_column

        except Exception as e:
            logging.error(f"Veri yükleme hatası: {str(e)}")
            raise e

    def process_data(self, df, text_column, quality_threshold=0.5, toxicity_threshold=0.5, batch_size=8):
        """
        Veriyi işler, kalite ve zararlılık skorlarını hesaplar.

        Args:
            df: İşlenecek veri çerçevesi
            text_column: Metin sütunu adı
            quality_threshold: Kalite eşik değeri
            toxicity_threshold: Zararlılık eşik değeri
            batch_size: İşlenecek grup boyutu

        Returns:
            pd.DataFrame: İşlenmiş veri
        """
        try:
            # Boş veya NaN değerli satırları kontrol et
            df = df.copy()
            df[text_column] = df[text_column].astype(str)
            df = df[df[text_column].str.strip() != ""]
            df = df.reset_index(drop=True)

            texts = df[text_column].tolist()

            # Kalite skorlarını hesapla
            if self.quality_scorer:
                logging.info("Kalite skorları hesaplanıyor...")
                quality_scores, quality_features = self.quality_scorer.batch_score(texts, batch_size=batch_size)
                df['quality_score'] = quality_scores

                # Kalite özelliklerini ekle
                for i, features in enumerate(quality_features):
                    for feat_name, feat_value in features.items():
                        if i == 0:  # İlk satır için sütun oluştur
                            df[feat_name] = np.nan
                        df.at[i, feat_name] = feat_value

            # Zararlılık skorlarını hesapla
            if self.toxicity_scorer:
                logging.info("Zararlılık skorları hesaplanıyor...")
                toxicity_scores = self.toxicity_scorer.batch_score(texts, batch_size=batch_size)
                df['toxicity_score'] = toxicity_scores

            # Eşik değerlerine göre etiketle
            if 'quality_score' in df.columns:
                df['low_quality'] = df['quality_score'] < quality_threshold

            if 'toxicity_score' in df.columns:
                df['is_toxic'] = df['toxicity_score'] > toxicity_threshold

            # Genel değerlendirme
            if 'quality_score' in df.columns and 'toxicity_score' in df.columns:
                df['acceptable'] = (df['quality_score'] >= quality_threshold) & (
                            df['toxicity_score'] <= toxicity_threshold)

            logging.info("Veri işleme tamamlandı.")
            return df

        except Exception as e:
            logging.error(f"Veri işleme hatası: {str(e)}")
            raise e

    def filter_data(self, df, quality_threshold=0.5, toxicity_threshold=0.5):
        """
        Veriyi belirlenen eşik değerlerine göre filtreler.

        Args:
            df: Filtrelenecek veri çerçevesi
            quality_threshold: Kalite eşik değeri
            toxicity_threshold: Zararlılık eşik değeri

        Returns:
            pd.DataFrame: Filtrelenmiş veri
        """
        filtered_df = df.copy()

        # Kalite filtreleme
        if 'quality_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['quality_score'] >= quality_threshold]

        # Zararlılık filtreleme
        if 'toxicity_score' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['toxicity_score'] <= toxicity_threshold]

        return filtered_df

    def save_data(self, df, output_path=None):
        """
        İşlenmiş veriyi kaydeder.

        Args:
            df: Kaydedilecek veri çerçevesi
            output_path: Çıktı dosyası yolu (belirtilmezse otomatik oluşturulur)

        Returns:
            str: Kaydedilen dosyanın yolu
        """
        if output_path is None:
            # Varsayılan çıktı yolunu oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "data/processed"

            # Klasörü oluştur
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"processed_data_{timestamp}.csv")

        # Veriyi kaydet
        df.to_csv(output_path, index=False)
        logging.info(f"Veri başarıyla kaydedildi: {output_path}")

        return output_path
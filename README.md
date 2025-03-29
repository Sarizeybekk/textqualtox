# Metin Kalitesi ve Zararlılık Değerlendirme 

## Proje Tanımı

Bu platform, metin verilerini otomatik olarak kalite ve zararlılık açısından değerlendirmek, analiz etmek ve iyileştirme önerileri sunmak amacıyla geliştirilmiştir. Türkçe metin analizi için özel olarak optimize edilmiş sistemimiz, metin değerlendirme, duygu analizi, dil tespiti ve anahtar kelime çıkarma gibi gelişmiş özellikleri barındırmaktadır.

## Özellikler

### Temel Özellikler

- **Metin Kalitesi Değerlendirmesi**: Dilbilgisi, tutarlılık, okunabilirlik gibi metriklerle metin kalitesini puanlar
- **Zararlılık Tespiti**: Metinlerdeki zararlı, saldırgan veya uygunsuz içeriği analiz eder
- **Toplu Dosya Analizi**: CSV ve Excel dosyalarını toplu olarak değerlendirir
- **Model Otomatik Seçimi**: En iyi performansı sağlayan modelleri otomatik olarak tespit eder

### Gelişmiş Özellikler

- **Duygu Analizi**: Metinlerin duygusal tonunu (pozitif, negatif, nötr) tespit eder
- **Metin İyileştirme**: Yazım hataları, dilbilgisi düzeltmeleri ve okunabilirlik önerileri sunar
- **Anahtar Kelime Çıkarma**: Metinde öne çıkan kelimeleri ve kavramları tespit eder
- **Dil Tespiti**: Metnin hangi dilde yazıldığını otomatik olarak belirler
- **İkili Kelime Grupları (Bigrams)**: Birlikte sık kullanılan kelime çiftlerini analiz eder.

##  Kullanılan Modeller

###  Zararlılık ve Duygu Analizi Modelleri

| Model | Açıklama | Dil | Kullanım |
|-------|----------|-----|----------|
| [`savasy/bert-base-turkish-sentiment`](https://huggingface.co/savasy/bert-base-turkish-sentiment) | Türkçe duygu analizi | 🇹🇷 | Birincil |
| [`dbmdz/bert-base-turkish-cased`](https://huggingface.co/dbmdz/bert-base-turkish-cased) | Genel Türkçe BERT | 🇹🇷 | Yedek |
| [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | İngilizce duygu analizi | 🇺🇸 | Yedek |

###  Kalite Değerlendirme Modelleri

| Model | Açıklama | Dil | Kullanım |
|-------|----------|-----|----------|
| [`sshleifer/distilbart-cnn-6-6`](https://huggingface.co/sshleifer/distilbart-cnn-6-6) | İngilizce özetleme | 🇺🇸 | Birincil |
| [`Helsinki-NLP/opus-mt-tr-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en) | TR-EN çeviri | 🇹🇷 | Yedek |
| [`sshleifer/distilbart-xsum-12-6`](https://huggingface.co/sshleifer/distilbart-xsum-12-6) | Hafif özetleme | 🇺🇸 | Yedek |

## 🚀 Deployment  
The application is live on **Hugging Face Spaces!** Try it here:  

👉 **[Veri Kalitesi ve Zararlılık Değerlendirme - Hugging Face](https://huggingface.co/spaces/sarizeybek/textqualtox)**  

## Kurulum

### Gereksinimler
- Python 3.10+
- Git
- 4GB+ RAM (8GB önerilir)
- (Opsiyonel) CUDA destekli GPU

### Kurulum Adımları

```bash
# 1. Repoyuu klonla
git clone https://github.com/Sarizeybekk/textqualtox.git
cd textqualtox

# 2. Sanal ortam oluştur
python -m venv venv

# 3. Sanal ortamı etkinleştir
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. Gerekli paketleri yükle
pip install -r requirements.txt

# 5. Uygulamayı çalıştır
streamlit run app.py

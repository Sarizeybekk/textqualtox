import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
from models.model_loader import ModelManager
from utils.quality_scorer import QualityScorer
from utils.toxicity_scorer import ToxicityScorer
from utils.data_handler import DataHandler
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.text_improver import TextImprover
from utils.keyword_extractor import KeywordExtractor
from utils.language_detector import LanguageDetector
import logging
import time
import re


# Loglama ayarları
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Veri klasörlerini oluştur
os.makedirs("data/processed", exist_ok=True)

# Sabitleri tanımla
SAMPLE_TEXT = """Bu bir örnek metindir. Bu metin, sistemin nasıl çalıştığını göstermek için kullanılmaktadır. 
Metin kalitesi ve zararlılık değerlendirmesi için kullanılabilir."""


def display_model_info(models_dict):
    """Model bilgilerini görüntüler"""
    model_info = models_dict.get("model_info", {})

    st.markdown("---")
    st.markdown("### Model Bilgileri")

    col1, col2 = st.columns(2)

    with col1:
        toxicity_info = model_info.get("toxicity", {})
        st.markdown("#### Zararlılık Modeli")

        model_name = toxicity_info.get("name", "Bilinmiyor")
        model_description = toxicity_info.get("description", "")
        model_language = toxicity_info.get("language", "")

        st.code(model_name, language="plaintext")


        language_icon = "🇹🇷" if model_language == "tr" else "🇺🇸" if model_language == "en" else "🌐"
        st.caption(f"{language_icon} {model_description}")

    with col2:
        quality_info = model_info.get("quality", {})
        st.markdown("#### Kalite Modeli")

        model_name = quality_info.get("name", "Bilinmiyor")
        model_description = quality_info.get("description", "")
        model_language = quality_info.get("language", "")

        st.code(model_name, language="plaintext")


        language_icon = "🇹🇷" if model_language == "tr" else "🇺🇸" if model_language == "en" else "🌐"
        st.caption(f"{language_icon} {model_description}")

    # Optimizasyon bilgisi
    st.info("""
    Bu sistem Türkçe metinler için otomatik optimize edilmiştir. 
    En iyi performansı gösteren modeller test sonuçlarına göre seçilmiştir.
    """)


@st.cache_resource
def load_models():
    """Modelleri yükler ve önbelleğe alır"""
    with st.spinner("Modeller değerlendiriliyor ve seçiliyor... Bu işlem birkaç dakika sürebilir."):
        # Örnek metinlerin bir kısmı
        sample_texts = [
            "Türkiye, zengin tarihi ve kültürel mirası ile dünyanın en etkileyici ülkelerinden biridir.",
            "turkiye guzel bi ulke. cok tarihi yerler var yani. denızleri guzel. yemekleride guzel.",
            "Bu grup insanlar gerçekten çok aptal! Hepsi geri zekalı ve cahil. Bunlarla konuşmak bile zaman kaybı.",
            "Kediler harika evcil hayvanlardır. Bağımsız yapıları vardır. Temizlik konusunda çok titizlerdir."
        ]

        try:
            # Cache dizinini belirle
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Gelişmiş model yükleme stratejisi
            model_manager = ModelManager(cache_dir=cache_dir, use_cache=True)
            success = model_manager.load_models_auto_select(sample_texts)

            if not success:
                st.error("Otomatik model seçimi başarısız oldu. Varsayılan modeller yükleniyor.")
                model_manager.load_default_models()

            models_dict = model_manager.get_models()

            toxicity_scorer = ToxicityScorer(
                model=models_dict["toxicity_model"],
                tokenizer=models_dict["toxicity_tokenizer"]
            )

            quality_scorer = QualityScorer(
                quality_pipeline=models_dict["quality_pipeline"]
            )

            # Skorlayıcıların model bilgilerini paylaşması için
            toxicity_scorer.model_info = models_dict["model_info"]["toxicity"]
            quality_scorer.model_info = models_dict["model_info"]["quality"]

            return toxicity_scorer, quality_scorer, models_dict

        except Exception as e:
            st.error(f"Model yükleme hatası: {str(e)}")
            # Yedek (basit) strateji
            logger.error(f"Model yükleme hatası: {str(e)}, basit modellere dönülüyor")

            toxicity_scorer = ToxicityScorer()  # Varsayılan modelle başlat
            quality_scorer = QualityScorer()  # Varsayılan modelle başlat

            models_dict = {
                "model_info": {
                    "toxicity": {"name": "Varsayılan Model",
                                 "description": "Hata nedeniyle varsayılan model kullanılıyor", "language": "unknown"},
                    "quality": {"name": "Varsayılan Model",
                                "description": "Hata nedeniyle varsayılan model kullanılıyor", "language": "unknown"}
                }
            }

            return toxicity_scorer, quality_scorer, models_dict


def analyze_single_text(text, toxicity_scorer, quality_scorer):
    """Tek bir metin için analiz yapar"""
    result = {}

    # Zararlılık analizi
    start_time = time.time()
    toxicity_score = toxicity_scorer.score_text(text)
    result["toxicity_score"] = toxicity_score
    result["toxicity_time"] = time.time() - start_time

    # Kalite analizi
    start_time = time.time()
    quality_score, quality_features = quality_scorer.score_text(text)
    result["quality_score"] = quality_score
    result["quality_features"] = quality_features
    result["quality_time"] = time.time() - start_time

    return result


def display_results(result, quality_threshold, toxicity_threshold):
    """Analiz sonuçlarını gösterir"""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kalite Puanı")
        quality_score = result["quality_score"]
        st.metric("Kalite", f"{quality_score:.2f}", delta=f"{quality_score - quality_threshold:.2f}")

        # Kalite özelliklerini görselleştir
        if "quality_features" in result:
            features = result["quality_features"]
            feature_df = pd.DataFrame({
                "Özellik": list(features.keys()),
                "Değer": list(features.values())
            })

            fig = px.bar(feature_df, x="Özellik", y="Değer", title="Kalite Özellikleri")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        st.info(f"Hesaplama süresi: {result['quality_time']:.2f} saniye")

    with col2:
        st.subheader("Zararlılık Puanı")
        toxicity_score = result["toxicity_score"]
        # Zararlılıkta düşük değer iyidir
        st.metric("Zararlılık", f"{toxicity_score:.2f}",
                  delta=f"{toxicity_threshold - toxicity_score:.2f}",
                  delta_color="inverse")

        # Zararlılık rengini göster
        fig, ax = plt.subplots(figsize=(3, 1))
        cmap = plt.cm.RdYlGn_r
        color = cmap(toxicity_score)
        ax.barh(0, toxicity_score, color=color)
        ax.barh(0, 1 - toxicity_score, left=toxicity_score, color="lightgrey")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        fig.tight_layout()
        st.pyplot(fig)

        st.info(f"Hesaplama süresi: {result['toxicity_time']:.2f} saniye")

    # Genel değerlendirme
    if quality_score >= quality_threshold and toxicity_score <= toxicity_threshold:
        st.success(" Bu metin kabul edilebilir kalite ve zararlılık seviyesindedir.")
    else:
        if quality_score < quality_threshold:
            st.warning(" Bu metnin kalitesi eşik değerin altındadır.")
        if toxicity_score > toxicity_threshold:
            st.error(" Bu metin kabul edilebilir zararlılık seviyesini aşmaktadır.")


def process_file(uploaded_file, toxicity_scorer, quality_scorer, quality_threshold, toxicity_threshold, batch_size):
    """Yüklenen dosyayı işler"""
    try:
        # Veri işleyici oluştur
        data_handler = DataHandler(quality_scorer, toxicity_scorer)

        # Veriyi yükle
        df, text_column = data_handler.load_data(uploaded_file)

        # İlerleme çubuğu göster
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Veriyi işle
        status_text.text("Veriler işleniyor...")

        # Veri işleme işlevini çağır
        processed_df = data_handler.process_data(
            df,
            text_column,
            quality_threshold,
            toxicity_threshold,
            batch_size
        )

        # İlerleme çubuğunu güncelle
        progress_bar.progress(100)
        status_text.text("İşlem tamamlandı!")

        return processed_df, text_column

    except Exception as e:
        st.error(f"Dosya işleme hatası: {str(e)}")
        logger.exception(f"Dosya işleme hatası: {str(e)}")
        return None, None


def main():
    """Ana uygulama işlevi"""
    st.set_page_config(page_title="📊 Veri Kalitesi ve Zararlılık Değerlendirme ",
     layout="wide")

    st.title(" 📊Veri Kalitesi ve Zararlılık Değerlendirme ")
    st.markdown("""
    Bu platform, metin verilerini otomatik olarak kalite ve zararlılık açısından değerlendirir.
    Tek bir metin veya CSV/Excel dosyası yükleyerek toplu analiz yapabilirsiniz.
    """)

    # Yan panel
    with st.sidebar:
        st.header("Ayarlar")

        st.subheader("Eşik Değerleri")
        quality_threshold = st.slider("Kalite Eşiği", 0.0, 1.0, 0.5,
                                      help="Bu değerin altındaki kalite skoruna sahip metinler düşük kaliteli olarak işaretlenir")
        toxicity_threshold = st.slider("Zararlılık Eşiği", 0.0, 1.0, 0.5,
                                       help="Bu değerin üzerindeki zararlılık skoruna sahip metinler zararlı olarak işaretlenir")

        st.subheader("Toplu İşleme")
        batch_size = st.slider("Grup Boyutu", 1, 32, 8,
                               help="Toplu değerlendirme için grup boyutu. Bellek sınırlamalarına göre ayarlayın.")

        # Modelleri yükle
        toxicity_scorer, quality_scorer, models_dict = load_models()

        # Model bilgilerini göster
        display_model_info(models_dict)

    # Sekmeleri oluştur
    tab1, tab2, tab3, tab4 = st.tabs(["Tek Metin Analizi", "Toplu Dosya Analizi",
                                      "Gelişmiş Metin Analizi", "Anahtar Kelime ve Dil Tespiti"])

    # Tek Metin Analizi sekmesi
    with tab1:
        st.subheader("Metin Analizi")

        text_input = st.text_area("Analiz edilecek metni girin:",
                                  value=SAMPLE_TEXT,
                                  height=150)

        if st.button("Analiz Et", type="primary", key="analyze_single"):
            if text_input and len(text_input.strip()) > 0:
                with st.spinner("Metin analiz ediliyor..."):
                    result = analyze_single_text(text_input, toxicity_scorer, quality_scorer)

                st.markdown("---")
                st.subheader("Analiz Sonuçları")
                display_results(result, quality_threshold, toxicity_threshold)
            else:
                st.error("Lütfen analiz için bir metin girin.")

    # Toplu Dosya Analizi sekmesi
    with tab2:
        st.subheader("Dosya Analizi")

        uploaded_file = st.file_uploader("CSV veya Excel dosyası yükleyin:",
                                         type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            # Dosya bilgilerini göster
            file_details = {
                "Dosya Adı": uploaded_file.name,
                "Dosya Boyutu": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write(file_details)

            # Dosyayı işle
            if st.button("Dosyayı İşle", type="primary", key="process_file"):
                with st.spinner("Dosya işleniyor... Bu işlem dosya boyutuna bağlı olarak biraz zaman alabilir."):
                    processed_df, text_column = process_file(
                        uploaded_file,
                        toxicity_scorer,
                        quality_scorer,
                        quality_threshold,
                        toxicity_threshold,
                        batch_size
                    )

                if processed_df is not None:
                    st.markdown("---")
                    st.subheader("İşlenmiş Veri")

                    # Veri önizlemesi göster
                    st.dataframe(processed_df.head(10), use_container_width=True)

                    # İstatistikler
                    st.markdown("### Özet İstatistikler")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Toplam Metin Sayısı", len(processed_df))

                    with col2:
                        acceptable_count = processed_df[
                            "acceptable"].sum() if "acceptable" in processed_df.columns else 0
                        acceptable_pct = acceptable_count / len(processed_df) * 100 if len(processed_df) > 0 else 0
                        st.metric("Kabul Edilebilir Metinler", f"{acceptable_count} ({acceptable_pct:.1f}%)")

                    with col3:
                        rejected_count = len(processed_df) - acceptable_count
                        rejected_pct = 100 - acceptable_pct
                        st.metric("Reddedilen Metinler", f"{rejected_count} ({rejected_pct:.1f}%)")

                    # Görselleştirmeler
                    st.markdown("### Veri Görselleştirme")
                    col1, col2 = st.columns(2)

                    with col1:
                        if "quality_score" in processed_df.columns:
                            fig = px.histogram(
                                processed_df,
                                x="quality_score",
                                nbins=20,
                                title="Kalite Skoru Dağılımı",
                                color_discrete_sequence=["#3366cc"]
                            )
                            fig.add_vline(x=quality_threshold, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        if "toxicity_score" in processed_df.columns:
                            fig = px.histogram(
                                processed_df,
                                x="toxicity_score",
                                nbins=20,
                                title="Zararlılık Skoru Dağılımı",
                                color_discrete_sequence=["#dc3912"]
                            )
                            fig.add_vline(x=toxicity_threshold, line_dash="dash", line_color="red")
                            st.plotly_chart(fig, use_container_width=True)

                    # Scatter plot
                    if "quality_score" in processed_df.columns and "toxicity_score" in processed_df.columns:
                        fig = px.scatter(
                            processed_df,
                            x="quality_score",
                            y="toxicity_score",
                            color="acceptable" if "acceptable" in processed_df.columns else None,
                            title="Kalite vs Zararlılık",
                            color_discrete_sequence=["#dc3912", "#3366cc"],
                            hover_data=[text_column]
                        )
                        fig.add_hline(y=toxicity_threshold, line_dash="dash", line_color="red")
                        fig.add_vline(x=quality_threshold, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

                    # Filtrelenmiş veri
                    data_handler = DataHandler(quality_scorer, toxicity_scorer)
                    filtered_df = data_handler.filter_data(processed_df, quality_threshold, toxicity_threshold)

                    st.markdown("### Filtrelenmiş Veri")
                    st.write(
                        f"Eşik değerlerini karşılayan {len(filtered_df)} metin ({len(filtered_df) / len(processed_df) * 100:.1f}%)")
                    st.dataframe(filtered_df.head(10), use_container_width=True)

                    # İndirme bağlantıları
                    st.markdown("### Verileri İndir")
                    col1, col2 = st.columns(2)

                    with col1:
                        # İşlenmiş veriyi CSV olarak dışa aktar
                        csv_processed = processed_df.to_csv(index=False)
                        st.download_button(
                            label="İşlenmiş Veriyi İndir (CSV)",
                            data=csv_processed,
                            file_name=f"processed_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )

                    with col2:
                        # Filtrelenmiş veriyi CSV olarak dışa aktar
                        csv_filtered = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Filtrelenmiş Veriyi İndir (CSV)",
                            data=csv_filtered,
                            file_name=f"filtered_{uploaded_file.name.split('.')[0]}.csv",
                            mime="text/csv"
                        )

    # Gelişmiş Metin Analizi sekmesi
    with tab3:
        st.subheader("Gelişmiş Metin Analizi")
        st.write("Bu sekmede duygu analizi ve metin iyileştirme önerilerini görebilirsiniz.")

        # Analizörler oluştur
        sentiment_analyzer = SentimentAnalyzer()
        text_improver = TextImprover()

        advanced_text_input = st.text_area("Analiz edilecek metni girin:",
                                           value=SAMPLE_TEXT,
                                           height=150,
                                           key="advanced_text")

        col1, col2 = st.columns(2)

        with col1:
            sentiment_option = st.checkbox("Duygu Analizi", value=True,
                                           help="Metnin duygusal tonunu analiz eder")

        with col2:
            improvement_option = st.checkbox("Metin İyileştirme", value=True,
                                             help="Metin kalitesini artırmak için öneriler sunar")

        if st.button("Gelişmiş Analiz Yap", type="primary", key="advanced_analyze"):
            if advanced_text_input and len(advanced_text_input.strip()) > 0:
                with st.spinner("Gelişmiş analiz yapılıyor..."):
                    # Duygu analizi
                    if sentiment_option:
                        sentiment_results = sentiment_analyzer.analyze_sentiment(advanced_text_input)

                        st.markdown("### Duygu Analizi Sonuçları")

                        # Duygu skoru ve tonu göster
                        sentiment_score = sentiment_results.get('score', 0)
                        dominant_sentiment = sentiment_results.get('dominant', 'neutral')

                        # Duygu tonuna göre renk ve emoji belirle
                        if sentiment_score > 0.2:
                            sentiment_color = "green"
                            sentiment_emoji = "😃"
                        elif sentiment_score < -0.2:
                            sentiment_color = "red"
                            sentiment_emoji = "😠"
                        else:
                            sentiment_color = "orange"
                            sentiment_emoji = "😐"

                        # Dominant duygu için Türkçe karşılık
                        sentiment_turkish = {
                            'positive': 'Pozitif',
                            'neutral': 'Nötr',
                            'negative': 'Negatif'
                        }.get(dominant_sentiment, 'Nötr')

                        st.markdown(f"**Duygu Tonu:** {sentiment_emoji} {sentiment_turkish}")
                        st.markdown(
                            f"**Duygu Skoru:** <span style='color:{sentiment_color}'>{sentiment_score:.2f}</span> (-1 ile 1 arasında)",
                            unsafe_allow_html=True)

                        # Duygu dağılımını göster
                        sentiment_df = pd.DataFrame({
                            'Duygu': ['Pozitif', 'Nötr', 'Negatif'],
                            'Skor': [
                                sentiment_results.get('positive', 0),
                                sentiment_results.get('neutral', 0),
                                sentiment_results.get('negative', 0)
                            ]
                        })

                        fig = px.bar(sentiment_df, x='Duygu', y='Skor',
                                     color='Duygu',
                                     color_discrete_map={'Pozitif': 'green', 'Nötr': 'gray', 'Negatif': 'red'},
                                     title="Duygu Dağılımı")
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)

                    # Metin iyileştirme
                    if improvement_option:
                        improvement_results = text_improver.improve_text(advanced_text_input)

                        st.markdown("### Metin İyileştirme Önerileri")

                        # Okunabilirlik göster
                        readability = improvement_results.get('readability', {'score': 0, 'level': 'bilinmiyor'})

                        # Okunabilirlik skoru için renk ve seviye belirleme
                        readability_score = readability.get('score', 0)
                        if readability_score >= 70:
                            readability_color = "green"
                        elif readability_score >= 50:
                            readability_color = "orange"
                        else:
                            readability_color = "red"

                        level_map = {
                            'çok_kolay': 'Çok Kolay',
                            'kolay': 'Kolay',
                            'orta_kolay': 'Orta-Kolay',
                            'orta': 'Orta',
                            'orta_zor': 'Orta-Zor',
                            'zor': 'Zor',
                            'çok_zor': 'Çok Zor',
                            'bilinmiyor': 'Bilinmiyor'
                        }
                        level_text = level_map.get(readability.get('level', 'bilinmiyor'), 'Bilinmiyor')

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Okunabilirlik Skoru", f"{readability_score:.1f}/100")
                            st.markdown(
                                f"**Okunabilirlik Seviyesi:** <span style='color:{readability_color}'>{level_text}</span>",
                                unsafe_allow_html=True)

                        with col2:
                            if 'avg_sentence_length' in readability:
                                st.metric("Ortalama Cümle Uzunluğu", f"{readability['avg_sentence_length']:.1f} kelime")

                        # İyileştirme önerileri
                        improvement_count = improvement_results.get('improvement_count', 0)
                        if improvement_count > 0:
                            st.markdown("#### Öneriler")
                            for i, suggestion in enumerate(improvement_results.get('suggestions', [])):
                                st.markdown(f"{i + 1}. {suggestion}")

                            # Düzeltilmiş metni göster
                            if improvement_results.get('corrected_text', '') != advanced_text_input:
                                st.markdown("#### Düzeltilmiş Metin")
                                st.code(improvement_results.get('improved_text', advanced_text_input), language=None)
                        else:
                            st.success("✓ Bu metin için iyileştirme önerisi bulunmamaktadır. Metin kalitesi iyi.")
            else:
                st.error("Lütfen analiz için bir metin girin.")

    # Anahtar Kelime ve Dil Tespiti sekmesi
    with tab4:
        st.subheader("Anahtar Kelime Çıkarma ve Dil Tespiti")
        st.write("Bu sekmede metninizin anahtar kelimelerini çıkarabilir ve dilini tespit edebilirsiniz.")

        # Analizörler oluştur
        keyword_extractor = KeywordExtractor()
        language_detector = LanguageDetector()

        # Metin giriş alanı
        keyword_text_input = st.text_area("Analiz edilecek metni girin:",
                                          value=SAMPLE_TEXT,
                                          height=150,
                                          key="keyword_text")

        # Ayarlar
        col1, col2, col3 = st.columns(3)

        with col1:
            keyword_method = st.selectbox(
                "Anahtar Kelime Metodu",
                ["combined", "tfidf", "textrank"],
                help="Anahtar kelime çıkarma algoritması"
            )

        with col2:
            num_keywords = st.slider(
                "Anahtar Kelime Sayısı",
                5, 20, 10,
                help="Çıkarılacak anahtar kelime sayısı"
            )

        with col3:
            detect_language = st.checkbox(
                "Dil Tespiti",
                value=True,
                help="Metnin dilini otomatik olarak tespit eder"
            )

        if st.button("Anahtar Kelime ve Dil Analizi", type="primary", key="keyword_analyze"):
            if keyword_text_input and len(keyword_text_input.strip()) > 0:
                with st.spinner("Analiz yapılıyor..."):
                    # Dil tespiti
                    if detect_language:
                        lang_result = language_detector.detect_language(keyword_text_input)

                        st.markdown("### Dil Tespiti Sonucu")

                        # Dil adı ve güven skoru
                        lang_code = lang_result['language_code']
                        lang_name = lang_result['language_name']
                        confidence = lang_result['confidence']

                        # Dil için bayrak ve renk
                        lang_flags = {
                            'tr': '🇹🇷',
                            'en': '🇺🇸',
                            'de': '🇩🇪',
                            'fr': '🇫🇷',
                            'es': '🇪🇸',
                            'unknown': '🌐'
                        }

                        lang_flag = lang_flags.get(lang_code, '🌐')

                        # Sonuçları göster
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"

                        st.markdown(f"### {lang_flag} Tespit Edilen Dil: {lang_name}")
                        st.markdown(f"**Güven Skoru:** <span style='color:{confidence_color}'>{confidence:.2f}</span>", unsafe_allow_html=True)

                        # Eğer tüm dil skorlarını göstermek istersek
                        if lang_result['scores']:
                            scores_df = pd.DataFrame({
                                'Dil': [language_detector.supported_languages.get(code, code) for code in lang_result['scores'].keys()],
                                'Kod': list(lang_result['scores'].keys()),
                                'Skor': list(lang_result['scores'].values())
                            })

                            # Skorlara göre sırala
                            scores_df = scores_df.sort_values(by='Skor', ascending=False).reset_index(drop=True)

                            # En yüksek skorlu diğer dillerin skorlarını göster
                            st.markdown("#### Dil Skorları")

                            # Geniş çubuk grafik
                            fig = px.bar(
                                scores_df.head(5),  # En yüksek 5 skoru göster
                                x='Skor',
                                y='Dil',
                                orientation='h',
                                title="Dil Algılama Skorları",
                                color='Skor',
                                color_continuous_scale='Viridis'
                            )

                            fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)

                    # Anahtar kelime çıkarma
                    keyword_results = keyword_extractor.extract_keywords(
                        keyword_text_input,
                        method=keyword_method,
                        num_keywords=num_keywords
                    )

                    st.markdown("### Anahtar Kelime Analizi")

                    # Kullanılan yöntemi göster
                    method_names = {
                        'tfidf': 'TF-IDF',
                        'textrank': 'TextRank',
                        'combined': 'Birleşik (TF-IDF + TextRank)'
                    }

                    st.write(f"Kullanılan yöntem: **{method_names.get(keyword_results['method'], keyword_results['method'])}**")

                    # Tekil anahtar kelimeleri göster
                    if keyword_results['keywords']:
                        keywords_df = pd.DataFrame({
                            'Anahtar Kelime': [kw[0] for kw in keyword_results['keywords']],
                            'Skor': [kw[1] for kw in keyword_results['keywords']]
                        })

                        # Skorlara göre sırala
                        keywords_df = keywords_df.sort_values(by='Skor', ascending=False).reset_index(drop=True)

                        # İki sütunlu düzen
                        col1, col2 = st.columns(2)

                        with col1:
                            # Anahtar kelime listesi
                            st.markdown("#### Anahtar Kelimeler")
                            for i, (keyword, score) in enumerate(zip(keywords_df['Anahtar Kelime'], keywords_df['Skor'])):
                                # Skora göre font boyutu ve kalınlığı ayarla
                                font_size = min(18, max(12, 12 + score * 6))
                                font_weight = "bold" if score > 0.6 else "normal"
                                st.markdown(
                                    f"<span style='font-size:{font_size}px; font-weight:{font_weight}'>{i+1}. {keyword}</span> <small>({score:.2f})</small>",
                                    unsafe_allow_html=True
                                )

                        with col2:
                            # Anahtar kelime grafiği
                            fig = px.bar(
                                keywords_df,
                                x='Anahtar Kelime',
                                y='Skor',
                                title="Anahtar Kelime Skorları",
                                color='Skor',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(xaxis={'categoryorder': 'total descending'})
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Metinde anahtar kelime bulunamadı.")

                    # İkili kelime gruplarını (bigram) göster
                    if keyword_results['bigrams']:
                        bigrams_df = pd.DataFrame({
                            'İkili Kelime Grubu': [bg[0] for bg in keyword_results['bigrams']],
                            'Skor': [bg[1] for bg in keyword_results['bigrams']]
                        })

                        st.markdown("#### İkili Kelime Grupları (Bigrams)")

                        # İkili kelimeleri tablo olarak göster
                        st.dataframe(bigrams_df, use_container_width=True)

                        # İkili kelime grafiği
                        fig = px.bar(
                            bigrams_df,
                            x='İkili Kelime Grubu',
                            y='Skor',
                            title="İkili Kelime Grubu Skorları",
                            color='Skor',
                            color_continuous_scale='Greens'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Metin içinde anahtar kelimeleri vurgula
                    if keyword_results['keywords']:
                        st.markdown("#### Anahtar Kelimeleri Vurgulanmış Metin")

                        highlighted_text = keyword_text_input
                        for keyword, _ in keyword_results['keywords']:
                            # Büyük/küçük harfe duyarsız olarak değiştirme yapmak için regex
                            pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
                            replacement = f"<mark><b>{keyword}</b></mark>"
                            highlighted_text = pattern.sub(replacement, highlighted_text)

                        st.markdown(highlighted_text, unsafe_allow_html=True)

                    # Özet bilgilendirme
                    st.success(f"Metinden toplam {len(keyword_results['keywords'])} anahtar kelime ve {len(keyword_results['bigrams'])} ikili kelime grubu çıkarıldı.")
            else:
                st.error("Lütfen analiz için bir metin girin.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Beklenmeyen bir hata oluştu: {str(e)}")
        logger.exception(f"Beklenmeyen bir hata: {str(e)}")
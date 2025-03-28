import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Metin içindeki anahtar kelimeleri çıkaran sınıf.
    TF-IDF, rakip kelimeler ve diğer metotlarla anahtar kelime çıkarma işlemi yapar.
    """

    def __init__(self):
        """Anahtar kelime çıkarıcıyı başlatır"""
        # Türkçe stopwords (durma kelimeleri)
        self.turkish_stopwords = [
            've', 'veya', 'ile', 'için', 'bu', 'bir', 'ya', 'de', 'da', 'ki', 'ne', 'her', 'çok',
            'daha', 'ama', 'fakat', 'lakin', 'ancak', 'gibi', 'kadar', 'sonra', 'önce', 'göre',
            'nasıl', 'neden', 'şey', 'ben', 'sen', 'o', 'biz', 'siz', 'onlar', 'kendi', 'aynı',
            'ise', 'mi', 'mı', 'mu', 'mü', 'hem', 'değil', 'hiç', 'olarak', 'evet', 'hayır',
            'belki', 'tüm', 'yani', 'hep', 'şu', 'şey', 'tabi', 'tamam', 'bunlar', 'şunlar',
            'böyle', 'öyle', 'şöyle', 'iki', 'üç', 'dört', 'beş', 'altı', 'yedi', 'sekiz', 'dokuz',
            'on', 'yüz', 'bin', 'milyon', 'milyar', 'var', 'yok', 'oldu', 'olur', 'oluyor', 'olacak'
        ]

        # İngilizce stopwords (durma kelimeleri)
        self.english_stopwords = [
            'the', 'and', 'a', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'by', 'from', 'about',
            'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during',
            'without', 'before', 'under', 'around', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
            'may', 'might', 'must', 'can', 'could', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'who', 'which', 'whose', 'whom', 'this', 'that', 'these',
            'those', 'am', 'is', 'are', 'was', 'were', 'an', 'my', 'your', 'his', 'its', 'our', 'their'
        ]

        # Tüm stopwords listesini birleştir
        self.stopwords = set(self.turkish_stopwords + self.english_stopwords)

        # TF-IDF vektörleyici
        self.tfidf_vectorizer = TfidfVectorizer(
            max_df=0.9,
            min_df=2,
            max_features=200,
            stop_words=self.stopwords,
            ngram_range=(1, 2)  # Tek kelimeler ve ikili kelime grupları
        )

        # Sayısal karakter ve noktalama işaretlerini temizlemek için regex pattern
        self.cleanup_pattern = re.compile(f'[{re.escape(string.punctuation)}]|[0-9]')

        logger.info("Anahtar kelime çıkarıcı başlatıldı")

    def preprocess_text(self, text):
        """
        Metni anahtar kelime çıkarma için ön işleme tabi tutar

        Args:
            text: İşlenecek metin

        Returns:
            str: Temizlenmiş metin
        """
        if not text:
            return ""

        # Küçük harfe çevir
        text = text.lower()

        # Noktalama işaretlerini temizle
        text = self.cleanup_pattern.sub(' ', text)

        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_keywords_tfidf(self, text, num_keywords=5):
        """
        TF-IDF kullanarak metinden anahtar kelimeleri çıkarır

        Args:
            text: Anahtar kelimeleri çıkarılacak metin
            num_keywords: Çıkarılacak anahtar kelime sayısı

        Returns:
            list: [(anahtar_kelime, skor), ...] formatında liste
        """
        try:
            if not text or len(text.strip()) < 10:
                return []

            # Metni ön işle
            processed_text = self.preprocess_text(text)

            # TF-IDF matrix oluştur
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])

            # Feature isimleri (kelimeler)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()

            # Kelimelerin TF-IDF skorlarını hesapla ve sırala
            tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
            sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)

            # En yüksek skorlu kelimeleri seç
            top_keywords = sorted_scores[:num_keywords]

            return top_keywords

        except Exception as e:
            logger.error(f"TF-IDF anahtar kelime çıkarma hatası: {str(e)}")
            return []

    def extract_keywords_textrank(self, text, num_keywords=5):
        """
        TextRank benzeri bir algoritma ile anahtar kelimeleri çıkarır

        Args:
            text: Anahtar kelimeleri çıkarılacak metin
            num_keywords: Çıkarılacak anahtar kelime sayısı

        Returns:
            list: [(anahtar_kelime, skor), ...] formatında liste
        """
        try:
            if not text or len(text.strip()) < 10:
                return []

            # Metni ön işle
            processed_text = self.preprocess_text(text)

            # Kelimeleri ayır
            words = processed_text.split()

            # Stopwords olmayan kelimeleri filtrele
            filtered_words = [word for word in words if word not in self.stopwords and len(word) > 2]

            # Kelime frekanslarını hesapla
            word_freq = Counter(filtered_words)

            # En sık geçen kelimeleri seç
            most_common = word_freq.most_common(num_keywords * 2)  # Daha fazla al, sonra filtreleyeceğiz

            # TF-IDF skorlaması ile benzer bir yaklaşım uygula
            # Kelime sıklığının logaritması * kelimenin benzersizliği
            scored_words = []
            for word, count in most_common:
                # Benzersizlik faktörü: Toplam kelime sayısı / kelimenin sıklığı
                uniqueness = len(filtered_words) / (count + 1)
                # Skor hesapla
                score = np.log(count + 1) * uniqueness
                scored_words.append((word, score))

            # Skorlara göre sırala
            scored_words.sort(key=lambda x: x[1], reverse=True)

            return scored_words[:num_keywords]

        except Exception as e:
            logger.error(f"TextRank anahtar kelime çıkarma hatası: {str(e)}")
            return []

    def extract_bigrams(self, text, num_bigrams=3):
        """
        Metinden ikili kelime gruplarını (bigram) çıkarır

        Args:
            text: Anahtar kelimeleri çıkarılacak metin
            num_bigrams: Çıkarılacak bigram sayısı

        Returns:
            list: [(bigram, skor), ...] formatında liste
        """
        try:
            if not text or len(text.strip()) < 10:
                return []

            # Metni ön işle
            processed_text = self.preprocess_text(text)

            # Bigram için vektörleyici
            bigram_vectorizer = CountVectorizer(
                ngram_range=(2, 2),
                stop_words=self.stopwords,
                max_features=100
            )

            # Bigram matrix oluştur
            bigram_matrix = bigram_vectorizer.fit_transform([processed_text])

            # Feature isimleri (bigramlar)
            feature_names = bigram_vectorizer.get_feature_names_out()

            # Bigramların skorlarını hesapla ve sırala
            bigram_scores = zip(feature_names, bigram_matrix.toarray()[0])
            sorted_scores = sorted(bigram_scores, key=lambda x: x[1], reverse=True)

            # En yüksek skorlu bigramları seç
            top_bigrams = sorted_scores[:num_bigrams]

            return top_bigrams

        except Exception as e:
            logger.error(f"Bigram çıkarma hatası: {str(e)}")
            return []

    def extract_keywords(self, text, method='combined', num_keywords=10):
        """
        Metinden anahtar kelimeleri çıkarır

        Args:
            text: Anahtar kelimeleri çıkarılacak metin
            method: Kullanılacak metod ('tfidf', 'textrank', 'combined')
            num_keywords: Toplam çıkarılacak anahtar kelime sayısı

        Returns:
            dict: {
                'keywords': [(anahtar_kelime, skor), ...],
                'bigrams': [(bigram, skor), ...],
                'method': kullanılan metod
            }
        """
        if not text or len(text.strip()) < 10:
            return {
                'keywords': [],
                'bigrams': [],
                'method': method
            }

        try:
            keywords = []

            if method == 'tfidf':
                keywords = self.extract_keywords_tfidf(text, num_keywords)
            elif method == 'textrank':
                keywords = self.extract_keywords_textrank(text, num_keywords)
            else:  # combined
                # TF-IDF ve TextRank sonuçlarını birleştir
                tfidf_keywords = self.extract_keywords_tfidf(text, num_keywords // 2)
                textrank_keywords = self.extract_keywords_textrank(text, num_keywords // 2)

                # İki listeyi birleştir
                combined = {}
                for keyword, score in tfidf_keywords + textrank_keywords:
                    if keyword in combined:
                        combined[keyword] = max(combined[keyword], score)
                    else:
                        combined[keyword] = score

                # En yüksek skorlu kelimeleri seç
                keywords = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:num_keywords]

            # Bigramları da ekle
            bigrams = self.extract_bigrams(text, num_keywords // 3)

            return {
                'keywords': keywords,
                'bigrams': bigrams,
                'method': method
            }

        except Exception as e:
            logger.error(f"Anahtar kelime çıkarma hatası: {str(e)}")
            return {
                'keywords': [],
                'bigrams': [],
                'method': method
            }
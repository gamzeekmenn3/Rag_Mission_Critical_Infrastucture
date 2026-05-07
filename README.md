# RAG-Based LLM System for Mission-Critical Infrastructure
---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlat
```bash
streamlit run app.py
```

Tarayıcıda otomatik olarak `http://localhost:8501` açılacaktır.

---

## 🏗️ Mimari Bileşenler

### 1. Data Ingestion (Veri Alımı)
- **Vektör Veritabanı:** Pinecone/Weaviate/Milvus benzeri yapı simüle edilmiştir
- **Chunking:** Semantic, Fixed-Size, Sentence-Based ve Paragraph-Based stratejiler
- **Embedding:** 128-dim'den 4096-dim'e kadar farklı boyutlar desteklenir

### 2. State Management (Durum Yönetimi)
- **Redis In-Memory:** Oturum bağlamı ve embedding cache katmanı
- **Contextual Consistency:** LLM'in geçmiş ve güncel veriler arasında çelişki yaşamasını önleyen senkronizasyon mekanizması
- **TTL Yönetimi:** Cache süresi dolduğunda otomatik yenileme

### 3. Fault Tolerance (Hata Toleransı)
- **Kafka Message Queue:** Asenkron mesaj kuyruğu simülasyonu
- **Dead Letter Queue (DLQ):** Başarısız mesajların kaybolmaması için yedek mekanizma
- **Fault Injection:** Network partition, timeout, rate limit gibi hataları simüle etme
- **Auto-Recovery:** Otomatik yeniden deneme mekanizması

### 4. Hallucination Prevention (Halüsinasyon Engelleme)
- **Retrieval Mekanizması:** Cosine similarity ile doğrulanmış bilgi çekimi
- **Similarity Threshold:** Düşük benzerlik skoru = yanıt reddi
- **Prompt Guard Rails:** Yalnızca sağlanan bağlam dahilinde yanıt zorunluluğu
- **Source Attribution:** Her yanıt için kaynak referansı

### 5. Performance Monitoring (Performans İzleme)
- **Latency Tracking:** Gerçek zamanlı gecikme ölçümü ve SLA sınırı
- **Accuracy Metrics:** Retrieval doğruluk takibi
- **Throughput Monitoring:** Dakika başına sorgu kapasitesi
- **Health Dashboard:** Tüm bileşenlerin sağlık durumu

---

## 📚 Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---------|-----------|
| UI Framework | Streamlit |
| Veri İşleme | NumPy, Pandas |
| Görselleştirme | Plotly |
| Vektör DB (Simüle) | FAISS (üretim için Pinecone/Milvus) |
| Cache Katmanı (Simüle) | Redis |
| Mesaj Kuyruğu (Simüle) | Apache Kafka / RabbitMQ |
| LLM (Üretim) | Claude / GPT-4o / LLaMA |

---

## 🎯 Demo Senaryoları

1. **RAG Query Engine** sekmesinde örnek sorgular gönderin
2. **Data Ingestion** sekmesinde yeni döküman yükleyin
3. **State Management** sekmesinde Redis cache simülasyonu görün
4. **Fault Tolerance** sekmesinde hata enjeksiyonu ve otomatik kurtarma test edin
5. **Performance Monitor** sekmesinde sistem metriklerini izleyin

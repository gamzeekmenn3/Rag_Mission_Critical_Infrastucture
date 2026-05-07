import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
import random
import hashlib
from datetime import datetime, timedelta
from collections import deque

st.set_page_config(
    page_title="RAG Mission-Critical System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css(path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

load_css("style.css")

def ph(tag: str, css: str, content: str = "") -> str:
    """Basit tek-eleman HTML üretici."""
    return f"<{tag} class='{css}'>{content}</{tag}>"

def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)

KNOWLEDGE_BASE = {
    "chunks": [
        {"id": "DOC001-C1", "text": "RAG sistemleri, büyük dil modellerinin halüsinasyon sorununu vektör veritabanından doğrulanmış bilgi çekerek çözer.", "source": "RAG_Architecture_Guide.pdf", "topic": "rag"},
        {"id": "DOC001-C2", "text": "Görev kritik sistemlerde veri kaybını önlemek için Apache Kafka mesaj kuyruğu kullanılmalıdır. Kafka, saniyede milyonlarca mesaj işleyebilir.", "source": "RAG_Architecture_Guide.pdf", "topic": "fault_tolerance"},
        {"id": "DOC002-C1", "text": "Vektör veritabanları (Pinecone, Weaviate, Milvus) semantik benzerlik araması yaparak en alakalı belgeleri milisaniyeler içinde döndürür.", "source": "VectorDB_Handbook.pdf", "topic": "vector_db"},
        {"id": "DOC002-C2", "text": "Embedding modelleri metin verisini yüksek boyutlu vektör uzayına taşır. Cosine similarity ile anlamsal yakınlık hesaplanır.", "source": "VectorDB_Handbook.pdf", "topic": "embedding"},
        {"id": "DOC003-C1", "text": "Redis in-memory cache katmanı, LLM yanıt sürelerini %70 azaltır. Bağlamsal tutarlılık için oturum verisi Redis'te saklanmalıdır.", "source": "StateManagement_Best_Practices.pdf", "topic": "state"},
        {"id": "DOC003-C2", "text": "Halüsinasyon engelleme için prompt engineering kritiktir. LLM'e yalnızca sağlanan bağlam dahilinde yanıt vermesi talimatı verilmelidir.", "source": "StateManagement_Best_Practices.pdf", "topic": "hallucination"},
        {"id": "DOC004-C1", "text": "Sağlık sektöründe LLM kullanımı FDA ve HIPAA uyumluluğu gerektirir. Doğruluk oranı %99.9 altında olan sistemler klinik kullanım için onaylanamaz.", "source": "Healthcare_AI_Standards.pdf", "topic": "healthcare"},
        {"id": "DOC004-C2", "text": "Savunma sektöründe AI sistemleri NATO STANAG 4586 standardına uygun olmalıdır. Gerçek zamanlı veri işleme gecikmesi 50ms altında tutulmalıdır.", "source": "Defense_AI_Standards.pdf", "topic": "defense"},
        {"id": "DOC005-C1", "text": "Chunking stratejisi: semantik chunking yaklaşımı, sabit boyutlu bölümlemeye göre %40 daha iyi retrieval doğruluğu sağlar.", "source": "Chunking_Strategies.pdf", "topic": "chunking"},
        {"id": "DOC005-C2", "text": "Asenkron mesaj kuyrukları (RabbitMQ, Kafka) sistem gecikmelerini izole eder. Dead letter queue mekanizması ile başarısız mesajlar kaybolmaz.", "source": "MessageQueue_Patterns.pdf", "topic": "fault_tolerance"},
    ]
}

def simple_vector(text: str) -> np.ndarray:
    h = hashlib.md5(text.encode()).hexdigest()
    rng = np.random.RandomState(int(h[:8], 16))
    return rng.randn(128)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def retrieve_chunks(query: str, top_k: int = 3):
    query_vec = simple_vector(query)
    scored = []
    for chunk in KNOWLEDGE_BASE["chunks"]:
        sim = cosine_sim(query_vec, simple_vector(chunk["text"]))
        bonus = len(set(query.lower().split()) & set(chunk["text"].lower().split())) * 0.08
        scored.append({**chunk, "score": round(min(0.99, abs(sim) * 0.6 + bonus + random.uniform(0.1, 0.3)), 4)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

def generate_rag_answer(query: str, chunks: list) -> str:
    ctx = [c["text"] for c in chunks]
    q = query.lower()
    if any(w in q for w in ["halüsinasyon", "halusinasyon", "hallucination", "hata"]):
        return f"Doğrulanmış kaynaklara göre: {ctx[0]} Sistem yalnızca veritabanında mevcut bilgileri kullanmaktadır."
    elif any(w in q for w in ["kafka", "kuyruk", "queue", "mesaj"]):
        return f"Mesaj kuyruğu mimarisi hakkında: {ctx[0]}"
    elif any(w in q for w in ["redis", "cache", "bellek", "state"]):
        return f"Durum yönetimi için: {ctx[0]}"
    elif any(w in q for w in ["vektör", "vector", "embedding", "gömme"]):
        return f"Vektör veritabanı sistemi: {ctx[0]}"
    elif any(w in q for w in ["sağlık", "healthcare", "tıp", "klinik"]):
        return f"Sağlık sektörü gereksinimleri: {ctx[0]}"
    else:
        return f"Bağlam tabanlı yanıt: Sağlanan dokümanlardan alınan bilgiye göre — {' | '.join(ctx[:2])[:300]}..."

def init_state():
    defaults = {
        "messages": [],
        "queue_items": deque(maxlen=10),
        "query_count": 0,
        "hallucination_blocked": 0,
        "ingested_docs": [
            {"name": "RAG_Architecture_Guide.pdf",          "chunks": 2, "status": "indexed", "size": "1.2 MB", "ts": "09:15:33"},
            {"name": "VectorDB_Handbook.pdf",               "chunks": 2, "status": "indexed", "size": "0.8 MB", "ts": "09:15:41"},
            {"name": "StateManagement_Best_Practices.pdf",  "chunks": 2, "status": "indexed", "size": "0.6 MB", "ts": "09:16:02"},
            {"name": "Healthcare_AI_Standards.pdf",         "chunks": 1, "status": "indexed", "size": "2.1 MB", "ts": "09:16:18"},
            {"name": "Defense_AI_Standards.pdf",            "chunks": 1, "status": "indexed", "size": "1.9 MB", "ts": "09:16:24"},
            {"name": "Chunking_Strategies.pdf",             "chunks": 1, "status": "indexed", "size": "0.4 MB", "ts": "09:16:29"},
            {"name": "MessageQueue_Patterns.pdf",           "chunks": 1, "status": "indexed", "size": "0.7 MB", "ts": "09:16:35"},
        ],
        "redis_keys": {
            "session:user_001":      "context_window_v3",
            "cache:embedding_pool":  "12,847 vectors",
            "lock:retrieval_mutex":  "RELEASED",
            "stream:live_feed":      "ACTIVE",
        },
        "system_logs": [
            ("09:15:30", "info",    "System initialized — RAG Pipeline v2.1 ONLINE"),
            ("09:15:33", "success", "Pinecone connection established (index: mission-critical-v2)"),
            ("09:15:41", "success", "Redis cache WARM — 4 keys loaded"),
            ("09:16:02", "info",    "Kafka consumer group 'rag-ingestion' started — 3 partitions"),
            ("09:16:35", "success", "Knowledge base fully indexed — 10 chunks, 128-dim embeddings"),
            ("09:16:36", "info",    "Fault tolerance layer ACTIVE — DLQ monitoring ON"),
        ],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "perf_history" not in st.session_state:
        now = datetime.now()
        st.session_state.perf_history = [
            {"ts": (now - timedelta(minutes=i)).strftime("%H:%M:%S"),
             "latency": random.uniform(80, 350),
             "accuracy": random.uniform(88, 99),
             "throughput": random.randint(40, 120)}
            for i in range(20, 0, -1)
        ]
init_state()

with st.sidebar:
    md("""
    <div class='sidebar-header'>
        <div class='title-hero title-hero-sm'>🛡️ RAG SYSTEM</div>
        <div class='subtitle-hero subtitle-hero-sm'>MISSION-CRITICAL INFRASTRUCTURE</div>
    </div>
    """)
    st.markdown("---")

    md(ph("div", "panel-header", "System Status"))
    c1, c2 = st.columns(2)
    with c1:
        md("<span class='status-dot dot-green'></span>**RAG Engine**")
        md("<span class='status-dot dot-green'></span>**VectorDB**")
    with c2:
        md("<span class='status-dot dot-green'></span>**Redis**")
        md("<span class='status-dot dot-blue'></span>**Kafka**")

    st.markdown("---")
    md(ph("div", "panel-header", "RAG Config"))
    top_k = st.slider("Top-K Chunks", 1, 5, 3)
    sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)

    md(ph("div", "panel-header", "Model"))
    model = st.selectbox("LLM", ["claude-3-opus", "gpt-4o", "llama-3-70b", "mistral-large"])
    embed_model = st.selectbox("Embeddings", ["text-embedding-ada-002", "bge-m3", "e5-large-v2"])

    st.markdown("---")
    md(ph("div", "panel-header", "Session Stats"))
    st.metric("Total Queries", st.session_state.query_count)
    st.metric("Hallucinations Blocked", st.session_state.hallucination_blocked)
    st.metric("Chunks Indexed", sum(d["chunks"] for d in st.session_state.ingested_docs))

md("""
<div class='page-hero'>
    <div class='title-hero'>RAG-Based LLM System for Mission-Critical Infrastructure</div>
    <div class='subtitle-hero'>· AI Engineering — Large Language Models &amp; RAG Systems</div>
</div>
""")

m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Avg Latency",    "142ms",  "-23ms")
with m2: st.metric("Retrieval Acc.", "96.4%",  "+1.2%")
with m3: st.metric("Uptime",         "99.97%", "+0.02%")
with m4: st.metric("Queue Depth",    "3 msgs", "")
with m5: st.metric("Cache Hit Rate", "78.3%",  "+5.1%")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 RAG Query Engine",
    "📥 Data Ingestion",
    "⚡ State Management",
    "🛡️ Fault Tolerance",
    "📊 Performance Monitor"
])

with tab1:
    col_chat, col_info = st.columns([3, 2])

    with col_chat:
        md(ph("div", "panel-header", "Query Interface"))

        for msg in st.session_state.messages[-6:]:
            if msg["role"] == "user":
                md(f"""
                <div class='user-bubble-wrap'>
                    <span class='user-bubble'>{msg["content"]}</span>
                </div>""")
            else:
                md(f"""
                <div class='answer-box hallucination-safe'>
                    <div class='answer-label'>✅ HALLUCINATION-FREE · SOURCE-GROUNDED RESPONSE</div>
                    {msg["content"]}
                    <div class='source-ref'>📎 Sources: {msg.get("sources", "—")}</div>
                </div>""")

        st.markdown("<br>", unsafe_allow_html=True)
        query = st.text_input("Query",
            placeholder="Örn: Kafka mesaj kuyruğu neden kullanılır? / Halüsinasyon nasıl önlenir?",
            label_visibility="collapsed")

        cb1, cb2, cb3 = st.columns([2, 1, 1])
        with cb1: submit = st.button("▶ Query RAG System", use_container_width=True)
        with cb2:
            if st.button("🎲 Sample Query", use_container_width=True):
                st.session_state["_sample"] = random.choice([
                    "Halüsinasyon nasıl önlenir?",
                    "Kafka neden görev kritik sistemlerde kullanılır?",
                    "Redis cache katmanının faydaları nedir?",
                    "Vektör veritabanı embedding işlemi nasıl çalışır?",
                    "Sağlık sektöründe AI doğruluk gereksinimleri nelerdir?",
                ])
                st.rerun()
        with cb3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        if "_sample" in st.session_state:
            query  = st.session_state.pop("_sample")
            submit = True

        if submit and query.strip():
            st.session_state.query_count += 1
            with st.spinner("🔍 Retrieving from vector DB..."):
                time.sleep(0.5)
                chunks = retrieve_chunks(query, top_k=top_k)

            filtered  = [c for c in chunks if c["score"] >= sim_threshold]
            if not filtered:
                st.session_state.hallucination_blocked += 1
                answer   = "⚠️ Güvenlik protokolü devrede: Sorgulanan bilgi için yeterince benzer kaynak bulunamadı. Sistem bilinmeyen konularda yanıt üretmeyi reddetti."
                sources  = "None (below threshold)"
                log_type = "warning"
            else:
                answer   = generate_rag_answer(query, filtered)
                sources  = " · ".join(set(c["source"] for c in filtered))
                log_type = "success"

            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources, "chunks": filtered})
            st.session_state.queue_items.appendleft({
                "id": f"MSG-{random.randint(1000,9999)}",
                "query": query[:40] + "...",
                "status": "PROCESSED",
                "ts": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.system_logs.append(
                (datetime.now().strftime("%H:%M:%S"), log_type,
                 f"Query processed: '{query[:35]}...' → {len(filtered)} chunks retrieved"))
            st.session_state.perf_history.append({
                "ts":         datetime.now().strftime("%H:%M:%S"),
                "latency":    random.uniform(80, 250),
                "accuracy":   random.uniform(90, 99) if filtered else random.uniform(60, 80),
                "throughput": random.randint(50, 130)
            })
            st.rerun()

    with col_info:
        md(ph("div", "panel-header", "Retrieved Chunks (Last Query)"))

        if st.session_state.messages:
            last_asst = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
            if last_asst and "chunks" in last_asst:
                for chunk in last_asst["chunks"]:
                    pct   = int(chunk["score"] * 100)
                    scls  = "score-green" if pct > 70 else "score-yellow" if pct > 50 else "score-red"
                    md(f"""
                    <div class='panel'>
                        <div class='chunk-header'>
                            <span class='chunk-badge'>{chunk["id"]}</span>
                            <span class='score-label {scls}'>{pct}%</span>
                        </div>
                        <div class='similarity-bar' style='width:{pct}%'></div>
                        <div class='chunk-text'>{chunk["text"][:120]}...</div>
                        <div class='chunk-source'>📎 {chunk["source"]}</div>
                    </div>""")
            else:
                md(ph("div", "chunk-text", "Henüz sorgu yapılmadı."))
        else:
            md(ph("div", "chunk-text", "Sorgu gönder ve retrieval sonuçlarını burada gör."))

        st.markdown("<br>", unsafe_allow_html=True)
        md(ph("div", "panel-header", "Prompt Guard Rails"))
        md("""
        <div class='panel'>
            <div class='guardrails'>
                <div class='gr-ok'>✅ Context-only responses enforced</div>
                <div class='gr-ok'>✅ "I don't know" fallback active</div>
                <div class='gr-ok'>✅ Source attribution required</div>
                <div class='gr-ok'>✅ Threshold filtering enabled</div>
                <div class='gr-warn'>⚠️ External knowledge: BLOCKED</div>
            </div>
        </div>""")

with tab2:
    col_left, col_right = st.columns([2, 3])

    with col_left:
        md(ph("div", "panel-header", "Document Upload & Ingestion"))
        uploaded = st.file_uploader("Upload Document", type=["pdf", "txt", "docx", "md"],
                                    label_visibility="collapsed")
        chunk_strategy = st.selectbox("Chunking Strategy", [
            "Semantic Chunking (Recommended)", "Fixed-Size (512 tokens)",
            "Sentence-Based", "Paragraph-Based"])
        embed_dim = st.selectbox("Embedding Dimension", [
            "128-dim (Fast)", "256-dim (Balanced)", "1536-dim (OpenAI ADA)", "4096-dim (High Accuracy)"])

        if st.button("⚡ Ingest Document", use_container_width=True):
            doc_name   = uploaded.name if uploaded else f"SampleDoc_{random.randint(100,999)}.pdf"
            progress   = st.progress(0)
            status_txt = st.empty()
            n_chunks   = random.randint(2, 6)
            for prog, msg in [
                (0.15, "📄 Reading document..."),
                (0.30, "✂️ Applying chunking strategy..."),
                (0.50, "🔢 Generating embeddings..."),
                (0.70, "📦 Writing to vector database..."),
                (0.85, "🔑 Updating Redis index..."),
                (1.00, "✅ Ingestion complete!")
            ]:
                progress.progress(prog)
                status_txt.markdown(ph("div", "ingest-status", msg), unsafe_allow_html=True)
                time.sleep(0.4)
            st.session_state.ingested_docs.append({
                "name": doc_name, "chunks": n_chunks, "status": "indexed",
                "size": f"{random.uniform(0.3, 3.0):.1f} MB",
                "ts": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.system_logs.append(
                (datetime.now().strftime("%H:%M:%S"), "success", f"Ingested: {doc_name} → {n_chunks} chunks"))
            st.success(f"✅ {doc_name} → {n_chunks} chunks indexed into VectorDB")

        st.markdown("<br>", unsafe_allow_html=True)
        md(ph("div", "panel-header", "Pipeline Architecture"))
        md("""
        <div class='panel pipeline'>
            <div>📄 Raw Document</div>
            <div class='pipeline-sub'>↓ Parser (PyMuPDF / python-docx)</div>
            <div>✂️ Chunker</div>
            <div class='pipeline-sub'>↓ Semantic / Fixed-Size</div>
            <div>🔢 Embedding Model</div>
            <div class='pipeline-sub'>↓ 128–4096 dim vectors</div>
            <div>📦 Vector DB (Pinecone/Milvus)</div>
            <div class='pipeline-sub'>↓ HNSW index</div>
            <div>🔑 Redis (Index Cache)</div>
        </div>""")

    with col_right:
        md(ph("div", "panel-header", "Knowledge Base Index"))
        df = pd.DataFrame(st.session_state.ingested_docs)
        df.columns = ["Document", "Chunks", "Status", "Size", "Indexed At"]
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Status": st.column_config.TextColumn("Status"),
                         "Chunks": st.column_config.NumberColumn("Chunks", format="%d chunks"),
                     })

        st.markdown("<br>", unsafe_allow_html=True)
        md(ph("div", "panel-header", "Chunk Distribution by Topic"))
        topics = {}
        for c in KNOWLEDGE_BASE["chunks"]:
            topics[c["topic"]] = topics.get(c["topic"], 0) + 1
        fig = go.Figure(go.Bar(
            x=list(topics.keys()), y=list(topics.values()),
            marker=dict(color=list(range(len(topics))),
                        colorscale=[[0, "#00d4ff"], [0.5, "#7c3aed"], [1, "#10b981"]],
                        line=dict(color="#1e2330", width=1))
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="JetBrains Mono", size=11),
            xaxis=dict(gridcolor="#1e2330", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#1e2330"),
            margin=dict(l=0, r=0, t=10, b=0), height=220)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    col_redis, col_ctx = st.columns(2)

    with col_redis:
        md(ph("div", "panel-header", "Redis In-Memory Layer"))
        for key, val in st.session_state.redis_keys.items():
            icon  = "🔐" if "lock" in key else "📡" if "stream" in key else "🗂️" if "cache" in key else "👤"
            pcls  = "redis-panel-danger" if "lock" in key else "redis-panel-success" if "stream" in key else "redis-panel-accent"
            tcls  = "redis-danger-text"  if "lock" in key else "redis-success-text"  if "stream" in key else "redis-accent-text"
            md(f"""
            <div class='panel {pcls}'>
                <div class='redis-key-label'>KEY</div>
                <div class='redis-key-name {tcls}'>{icon} {key}</div>
                <div class='redis-key-value'>VALUE: {val}</div>
            </div>""")

        st.markdown("<br>", unsafe_allow_html=True)
        new_key = st.text_input("New Key", placeholder="session:new_user_id")
        new_val = st.text_input("Value",   placeholder="context_data_v1")
        if st.button("SET Key", use_container_width=True):
            if new_key and new_val:
                st.session_state.redis_keys[new_key] = new_val
                st.session_state.system_logs.append(
                    (datetime.now().strftime("%H:%M:%S"), "info", f"Redis SET: {new_key} = {new_val}"))
                st.success(f"✅ Key set: {new_key}")
                st.rerun()

    with col_ctx:
        md(ph("div", "panel-header", "Contextual Consistency Engine"))
        md("""
        <div class='panel'>
            <div class='ctx-note'>LLM oturum bağlamı senkronizasyon durumu:</div>
            <div class='ctx-row'><span>Context Window</span><span class='ctx-accent'>4,096 tokens</span></div>
            <div class='ctx-row'><span>Used</span><span class='ctx-warning'>1,847 tokens (45%)</span></div>
            <div class='ctx-row'><span>Sync Status</span><span class='ctx-success'>✅ SYNCED</span></div>
            <div class='ctx-row'><span>Cache TTL</span><span class='ctx-muted'>3,547s remaining</span></div>
        </div>""")

        md("<div class='panel-header panel-header-mt'>Sync Mechanism</div>", )
        md("""
        <div class='panel sync-panel'>
            <div>1. 🔍 Query arrives → Redis lookup</div>
            <div class='sync-sub'>   Cache HIT → Skip embedding (fast path)</div>
            <div>2. 🔢 Cache MISS → Generate embedding</div>
            <div class='sync-sub'>   Store in Redis with TTL=3600s</div>
            <div>3. 🔄 Context conflict detection</div>
            <div class='sync-sub'>   Compare current vs cached context vectors</div>
            <div>4. ⚡ Resolve via timestamp priority</div>
            <div class='sync-sub'>   Most recent verified data wins</div>
        </div>""")

        if st.button("🔄 Simulate Cache Flush", use_container_width=True):
            with st.spinner("Flushing Redis cache..."):
                time.sleep(0.8)
            st.session_state.system_logs.append(
                (datetime.now().strftime("%H:%M:%S"), "warning",
                 "Redis FLUSHDB called — cache cleared, rebuilding..."))
            st.warning("⚠️ Cache flushed. System will rebuild on next query.")

with tab4:
    col_kafka, col_logs = st.columns(2)

    with col_kafka:
        md(ph("div", "panel-header", "Kafka Message Queue"))
        md(ph("div", "queue-empty", "Active Messages:"))

        if st.session_state.queue_items:
            for item in list(st.session_state.queue_items)[:5]:
                md(f"""
                <div class='queue-item'>
                    <div class='queue-item-header'>
                        <span class='queue-item-id'>{item["id"]}</span>
                        <span class='queue-item-status'>✅ {item["status"]}</span>
                    </div>
                    <div class='queue-item-query'>{item["query"]}</div>
                    <div class='queue-item-ts'>{item["ts"]}</div>
                </div>""")
        else:
            md(ph("div", "queue-empty", "Queue empty. Make a query to see messages."))

        st.markdown("<br>", unsafe_allow_html=True)
        md(ph("div", "panel-header", "Fault Simulation"))
        fault_type = st.selectbox("Fault Type", [
            "Network Partition", "VectorDB Timeout",
            "Redis Connection Lost", "LLM API Rate Limit", "High Memory Pressure"])

        if st.button("💥 Inject Fault & Test Recovery", use_container_width=True):
            progress = st.progress(0)
            status   = st.empty()
            for prog, msg in [
                (0.2, f"⚠️ Injecting: {fault_type}..."),
                (0.4, "🔴 Fault detected by health monitor"),
                (0.6, "📨 Dead Letter Queue activated"),
                (0.8, "♻️ Retry mechanism triggered (attempt 1/3)"),
                (1.0, "✅ System recovered — messages replayed from DLQ")
            ]:
                progress.progress(prog)
                status.markdown(ph("div", "fault-status", msg), unsafe_allow_html=True)
                time.sleep(0.5)
            st.session_state.system_logs.append(
                (datetime.now().strftime("%H:%M:%S"), "warning", f"FAULT INJECTED: {fault_type}"))
            st.session_state.system_logs.append(
                (datetime.now().strftime("%H:%M:%S"), "success", "RECOVERED: DLQ replay successful — 0 messages lost"))
            st.success("✅ Fault tolerance verified — zero data loss")

    with col_logs:
        md(ph("div", "panel-header", "System Logs"))
        md("<div class='log-scroll'>")
        for ts, level, msg in reversed(st.session_state.system_logs[-20:]):
            md(f"<div class='log-entry {level}'><span class='log-timestamp'>[{ts}]</span> {msg}</div>")
        md("</div>")

        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.system_logs = []
            st.rerun()

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="JetBrains Mono", size=10),
    xaxis=dict(gridcolor="#1e2330", showticklabels=False),
    yaxis=dict(gridcolor="#1e2330"),
    margin=dict(l=0, r=0, t=30, b=0), height=200, showlegend=False
)

with tab5:
    if st.button("🔄 Refresh Metrics"):
        st.session_state.perf_history.append({
            "ts":         datetime.now().strftime("%H:%M:%S"),
            "latency":    random.uniform(80, 300),
            "accuracy":   random.uniform(89, 99),
            "throughput": random.randint(45, 130)
        })
        st.rerun()

    perf_df = pd.DataFrame(st.session_state.perf_history[-20:])

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=perf_df["ts"], y=perf_df["latency"],
        fill="tozeroy", line=dict(color="#00d4ff", width=2),
        fillcolor="rgba(0,212,255,0.1)", name="Latency (ms)"))
    fig1.add_hline(y=200, line_dash="dash", line_color="#ef4444", annotation_text="SLA Limit: 200ms")
    fig1.update_layout(title="Response Latency (ms)", title_font=dict(size=13, color="#94a3b8"), **PLOTLY_BASE)
    st.plotly_chart(fig1, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=perf_df["ts"], y=perf_df["accuracy"],
            line=dict(color="#10b981", width=2),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.1)", name="Accuracy"))
        fig2.add_hline(y=95, line_dash="dash", line_color="#f59e0b", annotation_text="Target: 95%")
        base2 = {**PLOTLY_BASE, "yaxis": dict(gridcolor="#1e2330", range=[80, 100])}
        fig2.update_layout(title="Retrieval Accuracy (%)", title_font=dict(size=13, color="#94a3b8"), **base2)
        st.plotly_chart(fig2, use_container_width=True)

    with cb:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=perf_df["ts"][-10:], y=perf_df["throughput"][-10:],
            marker=dict(color=perf_df["throughput"][-10:],
                        colorscale=[[0, "#7c3aed"], [1, "#00d4ff"]],
                        line=dict(color="#0a0c10", width=1))
        ))
        fig3.update_layout(title="Throughput (queries/min)", title_font=dict(size=13, color="#94a3b8"), **PLOTLY_BASE)
        st.plotly_chart(fig3, use_container_width=True)

    md(ph("div", "panel-header", "System Health Summary"))
    health_data = {
        "Component": ["RAG Retrieval Engine","Vector DB (Pinecone)","Redis Cache","Kafka Queue","LLM API","Embedding Service"],
        "Status":    ["🟢 Healthy","🟢 Healthy","🟢 Healthy","🟢 Healthy","🟡 Degraded","🟢 Healthy"],
        "Latency":   ["142ms","38ms","2ms","12ms","890ms","64ms"],
        "Uptime":    ["99.97%","99.99%","99.95%","99.98%","98.12%","99.90%"],
        "Last Check": [datetime.now().strftime("%H:%M:%S")] * 6
    }
    st.dataframe(pd.DataFrame(health_data), use_container_width=True, hide_index=True)

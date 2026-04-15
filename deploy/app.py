from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import csv
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

app = Flask(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/datacorpus.json"
LOG_CSV     = "history.csv"

EMBEDDING_MODELS = {
    # "Indo Sentence BERT": {
    #     "name":    "firqaaa/indo-sentence-bert-base",
    #     "db_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/faiss_indexeindobert",
    #     "description": "Optimized for Indonesian text"
    # },
    "BGE-M3": {
        "name":    "/Users/muhammadzuamaalamin/Documents/fintunellm/model/bge-m3",
        "db_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/faiss_indexbgem3",
        "description": "Multilingual, high accuracy"
    },
    # "Multilingual E5 Small": {
    #     "name":    "/Users/muhammadzuamaalamin/Documents/fintunellm/model/multilingual-e5-small",
    #     "db_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/rag/gemma/faiss_indexe5",
    #     "description": "Lightweight & fast"
    # },
    # "Multilingual All Mini LM": {
    #     "name":    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    #     "db_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/rag/gemma/faiss_indexallminiLm",
    #     "description": "Lightweight & fast"
    # },
        "E5-sample-Law-Indo": {
        "name":    "mzuama/E5-sampel-law",
        "db_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/faiss_indexe5lawsampel",
        "description": "Multilingual, high accuracy"
    }

}

# Fallback jika Ollama tidak bisa dijangkau
DEFAULT_LLM_MODELS = [
    # {"name": "gemma3:4b",   "description": "-"},
    {"name": "ministral-3:3b",  "description": "-"},
    # {"name": "qwen3:4b", "description": "-"},
    # {"name": "qwen3:8b",  "description": "-"},
    # {"name": "qwen2.5:7b",  "description": "-"},
    # {"name": "gemma2:2b",  "description": "-"},
    {"name": "ministral-3:8b",  "description": "-"},
    {"name": "gemma4:e2b",  "description": "-"},
    {"name": "gemma4:e4b",  "description": "-"}
]

PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
Kamu adalah asisten hukum Indonesia yang membantu menjelaskan aturan KUHP dengan bahasa yang mudah dipahami.

Konteks (kutipan pasal KUHP):
{context}

Pertanyaan:
{question}

Instruksi:
- Jawab pertanyaan berdasarkan konteks di atas.
- Gunakan bahasa yang jelas dan mudah dipahami.
- Saat menyebut suatu aturan, SELALU cantumkan nomor pasal DAN kutipan singkat isi pasalnya.
- Format referensi pasal: **Pasal [nomor]**: "[kutipan singkat isi pasal]"
- Jika konteks tidak relevan, jawab: "Informasi tidak ditemukan dalam dokumen."

Contoh:
Pertanyaan: Apakah menghina orang di depan umum bisa dipidana?
Jawaban: Ya, menghina orang secara lisan di depan umum dapat dikenakan pidana.

Dasar hukum:
- **Pasal 433**: "Barang siapa dengan sengaja menyerang kehormatan atau nama baik seseorang dengan lisan di muka umum, diancam dengan pidana penjara paling lama 9 bulan."
- **Pasal 441**: "Jika penghinaan dilakukan dengan tulisan atau gambar, diancam dengan pidana penjara paling lama 1 tahun 6 bulan."

Jawaban:
""")
# ─── Global cache ───────────────────────────────────────────────────────────────
_documents    = None
_retrievers   = {}
 
def load_documents():
    global _documents
    if _documents is not None:
        return _documents
    data = pd.read_json(DATA_PATH)
    data = data.drop(columns=[c for c in ['id','pasal','bab','judul','ayat','buku','bagian','paragraf'] if c in data.columns], errors='ignore')
    data['context'] = (data['context']
        .str.replace("\n", " ")
        .str.replace(r" +", " ", regex=True)
        .str.strip())
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=50)
    docs = []
    for pasal in data['context']:
        if len(pasal) > 4000:
            docs.extend([Document(page_content=s) for s in splitter.split_text(pasal)])
        else:
            docs.append(Document(page_content=pasal))
    _documents = docs
    return docs
 
def get_retriever(model_key: str):
    if model_key in _retrievers:
        return _retrievers[model_key]
    cfg  = EMBEDDING_MODELS[model_key]
    docs = load_documents()
    emb  = HuggingFaceEmbeddings(model_name=cfg["name"], model_kwargs={"device": "cpu"})
    db_path = cfg["db_path"]
    if not os.path.exists(db_path):
        vs = FAISS.from_documents(docs, emb)
        vs.save_local(db_path)
    else:
        vs = FAISS.load_local(db_path, emb, allow_dangerous_deserialization=True)
    bm25   = BM25Retriever.from_documents(docs); bm25.k = 5
    hybrid = EnsembleRetriever(
        retrievers=[bm25, vs.as_retriever(search_kwargs={"k": 5})],
        weights=[0.5, 0.5]
    )
    _retrievers[model_key] = hybrid
    return hybrid
 
def log_to_csv(question, answer, embed_key, llm_model):
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model_embedding", "model_llm", "question", "answer"])
        writer.writerow([datetime.now().isoformat(), embed_key, llm_model, question, answer])
 
def get_ollama_models():
    """Fetch model list from Ollama API, fallback to defaults on error."""
    try:
        r = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.ok:
            raw = r.json().get("models", [])
            return [{"name": m["name"], "description": f"{round(m.get('size',0)/1e9, 1)} GB"} for m in raw]
    except Exception:
        pass
    return DEFAULT_LLM_MODELS
 
# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", models=EMBEDDING_MODELS, llm_models=get_ollama_models())
 
@app.route("/llm-models")
def llm_models_route():
    return jsonify(get_ollama_models())
 
@app.route("/ask", methods=["POST"])
def ask():
    data      = request.get_json()
    question  = (data.get("question") or "").strip()
    embed_key = data.get("model_key", "")
    llm_model = (data.get("llm_model") or "gemma3:4b").strip()
    if not question:
        return jsonify({"error": "Pertanyaan tidak boleh kosong."}), 400
    if embed_key not in EMBEDDING_MODELS:
        return jsonify({"error": "Model embedding tidak valid."}), 400
    try:
        retriever = get_retriever(embed_key)
        llm       = ChatOllama(model=llm_model, temperature=0.1)
        qa_chain  = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT_TEMPLATE},
            return_source_documents=True
        )
        response = qa_chain.invoke({"query": question})
        answer   = response["result"]
        log_to_csv(question, answer, embed_key, llm_model)
        return jsonify({"answer": answer, "model_key": embed_key, "llm_model": llm_model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
@app.route("/history")
def history():
    if not os.path.exists(LOG_CSV):
        return jsonify({"rows": []})
    df   = pd.read_csv(LOG_CSV)
    rows = df.to_dict(orient="records")
    return jsonify({"rows": rows})
 
@app.route("/download-csv")
def download_csv():
    if not os.path.exists(LOG_CSV):
        return jsonify({"error": "Belum ada riwayat."}), 404
    return send_file(LOG_CSV, as_attachment=True, download_name="riwayat_pertanyaan.csv")
 
@app.route("/models")
def models():
    return jsonify(list(EMBEDDING_MODELS.keys()))
if __name__ == "__main__":
    app.run(debug=True, port=5000)
"""
Evaluasi Retrieval: Precision@K · Recall@K · MRR · nDCG@K
==========================================================
Mengevaluasi tiga metode retrieval (BM25, TF-IDF, Semantic)
menggunakan ground truth berbasis keyword matching pada UUD 1945.

Output:
  - Tabel per-query di terminal
  - eval_results.json  → untuk viewer HTML
"""

import json
import logging
import math
import re
from pathlib import Path
from langchain.retrievers import EnsembleRetriever
import numpy as np
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

CONFIG = {
    "json_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/datacorpus.json",
    "embedding_model": "/Users/muhammadzuamaalamin/Documents/fintunellm/model/bge-m3",
    "faiss_index_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/faiss_indexbgem3",
    "ground_truth_path": "/Users/muhammadzuamaalamin/Documents/RisetTextMining/retrieval2/datatest22.json",
    "output_json": "eval_results.json",
    "chunk_size": 5000,
    "chunk_overlap": 50,
    "top_k": 10,
    "hybrid_weights": (0.5, 0.5),
}

# ---------------------------------------------------------------------------
# Relevance judge: apakah sebuah chunk relevan untuk query tertentu?
# Strategi: chunk dianggap relevan jika mengandung ≥1 keyword dari ground truth.
# ---------------------------------------------------------------------------

def is_relevant(chunk_text: str, relevant_keywords: list[str]) -> bool:
    """
    Cek relevansi berdasarkan substring matching (case-insensitive).
    Chunk dianggap relevan jika mengandung setidaknya satu keyword relevan.
    """
    text_lower = chunk_text.lower()
    for kw in relevant_keywords:
        # Normalisasi: hapus tanda baca berlebih, lowercase
        kw_clean = kw.lower().strip()
        if kw_clean in text_lower:
            return True
    return False


def get_relevance_labels(retrieved_chunks: list[str], relevant_keywords: list[str]) -> list[int]:
    """Kembalikan list binary [1, 0, 1, ...] untuk setiap chunk yang di-retrieve."""
    return [1 if is_relevant(c, relevant_keywords) else 0 for c in retrieved_chunks]

# ---------------------------------------------------------------------------
# Metrik Evaluasi
# ---------------------------------------------------------------------------

def precision_at_k(labels: list[int], k: int) -> float:
    """
    Precision@K = (jumlah relevan dalam top-K) / K
    Mengukur: dari K dokumen yang diambil, berapa persen yang relevan?
    """
    top_k = labels[:k]
    return sum(top_k) / k if k > 0 else 0.0


def recall_at_k(labels: list[int], relevant_keywords: list[str], k: int) -> float:
    """
    Recall@K = (jumlah relevan dalam top-K) / (total dokumen relevan yang ADA)
    Catatan: 'total relevan' di sini diestimasi dari jumlah keyword unik ground truth,
    karena kita tidak tahu persis berapa chunk dalam corpus yang relevan.
    Pendekatan: anggap 1 chunk relevan per keyword → total_relevant = len(relevant_keywords)
    """
    top_k = labels[:k]
    total_relevant = len(relevant_keywords)  # estimasi konservatif
    if total_relevant == 0:
        return 0.0
    return sum(top_k) / total_relevant


def reciprocal_rank(labels: list[int]) -> float:
    """
    Reciprocal Rank = 1 / rank_dokumen_relevan_pertama
    Jika tidak ada dokumen relevan → 0.
    Mengukur: seberapa tinggi dokumen relevan pertama muncul dalam ranking.
    """
    for i, label in enumerate(labels, 1):
        if label == 1:
            return 1.0 / i
    return 0.0


def ndcg_at_k(labels: list[int], k: int) -> float:
    """
    nDCG@K = DCG@K / IDCG@K
    DCG menggunakan graded relevance (binary di sini: 0 atau 1).
    nDCG = 1 berarti urutan sempurna (semua relevan di atas).
    """
    def dcg(rels, k):
        return sum(
            rel / math.log2(i + 2)
            for i, rel in enumerate(rels[:k])
        )

    actual_dcg = dcg(labels, k)
    # IDCG: semua yang relevan di posisi teratas
    ideal_labels = sorted(labels, reverse=True)
    ideal_dcg = dcg(ideal_labels, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def average_precision(labels: list[int]) -> float:
    """
    Average Precision (AP) = rata-rata Precision@k hanya di posisi relevan.
    MAP = mean dari AP semua query.
    """
    hits = 0
    sum_precision = 0.0
    for i, label in enumerate(labels, 1):
        if label == 1:
            hits += 1
            sum_precision += hits / i
    total_relevant = sum(labels)
    return sum_precision / total_relevant if total_relevant > 0 else 0.0


def compute_all_metrics(labels: list[int], relevant_keywords: list[str], k: int) -> dict:
    return {
        "precision_at_k":  round(precision_at_k(labels, k), 4),
        "recall_at_k":     round(recall_at_k(labels, relevant_keywords, k), 4),
        "mrr":             round(reciprocal_rank(labels), 4),
        "ndcg_at_k":       round(ndcg_at_k(labels, k), 4),
        "ap":              round(average_precision(labels), 4),
        "labels":          labels,
        "hits":            sum(labels[:k]),
    }

# ---------------------------------------------------------------------------
# Retriever wrappers (sama seperti comparison script)
# ---------------------------------------------------------------------------

class BM25RetrieverWrapper:
    def __init__(self, chunks, k):
        self.ret = BM25Retriever.from_documents(chunks)
        self.ret.k = k

    def retrieve_texts(self, query: str) -> list[str]:
        return [d.page_content for d in self.ret.invoke(query)]


class TFIDFRetriever:
    def __init__(self, chunks, k):
        self.k = k
        self.chunks = chunks
        self.texts = [c.page_content for c in chunks]
        self.vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.mat = self.vec.fit_transform(self.texts)

    def retrieve_texts(self, query: str) -> list[str]:
        qv = self.vec.transform([query])
        scores = cosine_similarity(qv, self.mat).flatten()
        top_idx = np.argsort(scores)[::-1][: self.k]
        return [self.texts[i] for i in top_idx]


class SemanticRetriever:
    def __init__(self, chunks, cfg):
        self.k = cfg["top_k"]
        emb = HuggingFaceEmbeddings(
            model_name=cfg["embedding_model"],
            model_kwargs={"device": "cpu"},
        )
        fp = cfg["faiss_index_path"]
        if Path(fp).exists():
            self.vs = FAISS.load_local(fp, emb, allow_dangerous_deserialization=True)
        else:
            self.vs = FAISS.from_documents(chunks, emb)
            self.vs.save_local(fp)

    def retrieve_texts(self, query: str) -> list[str]:
        results = self.vs.similarity_search_with_score(query, k=self.k)
        return [doc.page_content for doc, _ in results]



class HybridRetriever:
    """BM25 + Semantic dengan EnsembleRetriever (weighted)."""
    def __init__(self, chunks, cfg, weights=(0.5, 0.5)):
        k = cfg["top_k"]

        bm25 = BM25Retriever.from_documents(chunks)
        bm25.k = k

        emb = HuggingFaceEmbeddings(
            model_name=cfg["embedding_model"],
            model_kwargs={"device": "cpu"},
        )
        fp = cfg["faiss_index_path"]
        vs = FAISS.load_local(fp, emb, allow_dangerous_deserialization=True) \
            if Path(fp).exists() \
            else FAISS.from_documents(chunks, emb)

        self.ensemble = EnsembleRetriever(
            retrievers=[bm25, vs.as_retriever(search_kwargs={"k": k})],
            weights=list(weights),
        )
        self.k = k

    def retrieve_texts(self, query: str) -> list[str]:
        return [d.page_content for d in self.ensemble.invoke(query)][: self.k]
# ---------------------------------------------------------------------------
# Pipeline evaluasi utama
# ---------------------------------------------------------------------------

def load_chunks(cfg: dict) -> list:
    loader = JSONLoader(file_path=cfg["json_path"], jq_schema=".[]", text_content=False)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Total chunks: %d", len(chunks))
    return chunks


def run_evaluation(cfg: dict) -> dict:
    # Load data
    chunks = load_chunks(cfg)
    with open(cfg["ground_truth_path"], encoding="utf-8") as f:
        ground_truth = json.load(f)

    k = cfg["top_k"]

    # Build retrievers
    logger.info("Membangun BM25...")
    bm25 = BM25RetrieverWrapper(chunks, k)
    logger.info("Membangun TF-IDF...")
    tfidf = TFIDFRetriever(chunks, k)
    logger.info("Membangun Semantic (FAISS)...")
    semantic = SemanticRetriever(chunks, cfg)
    logger.info("Membangun Hybrid (BM25 + Semantic)...")
    hybrid = HybridRetriever(chunks, cfg, weights=(0.5, 0.5))

    retrievers = {
        "BM25":     bm25,
        "TF-IDF":   tfidf,
        "Semantic": semantic,
        "Hybrid":   hybrid,
    }

    # Struktur hasil
    results = {
        method: {
            "per_query": [],
            "aggregate": {}
        }
        for method in retrievers
    }

    # ── Evaluasi per query ────────────────────────────────────────────────
    for item in ground_truth:
        qid     = item["query_id"]
        query   = item["query"]
        rel_kws = item["relevant_chunks_keywords"]

        logger.info("[%s] %s", qid, query)

        for method_name, retriever in retrievers.items():
            retrieved_texts = retriever.retrieve_texts(query)
            labels          = get_relevance_labels(retrieved_texts, rel_kws)
            metrics         = compute_all_metrics(labels, rel_kws, k)

            # Simpan detail per chunk untuk viewer
            chunks_detail = []
            for rank, (text, label) in enumerate(zip(retrieved_texts, labels), 1):
                chunks_detail.append({
                    "rank":     rank,
                    "content":  text,
                    "relevant": bool(label),
                    "matched_keywords": [
                        kw for kw in rel_kws if kw.lower() in text.lower()
                    ],
                })

            results[method_name]["per_query"].append({
                "query_id":           qid,
                "query":              query,
                "relevant_keywords":  rel_kws,
                "notes":              item.get("notes", ""),
                "metrics":            metrics,
                "retrieved_chunks":   chunks_detail,
            })

    # ── Aggregate (mean across queries) ──────────────────────────────────
    for method_name in retrievers:
        per_q = results[method_name]["per_query"]
        agg = {}
        for metric_key in ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "ap"]:
            vals = [q["metrics"][metric_key] for q in per_q]
            agg[f"mean_{metric_key}"] = round(float(np.mean(vals)), 4)
            agg[f"std_{metric_key}"]  = round(float(np.std(vals)),  4)
        agg["map"] = agg.pop("mean_ap")  # MAP = mean AP
        results[method_name]["aggregate"] = agg

    # ── Simpan JSON ───────────────────────────────────────────────────────
    output = {
        "config": {"top_k": k, "num_queries": len(ground_truth)},
        "methods": list(retrievers.keys()),
        "results": results,
    }
    with open(cfg["output_json"], "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info("Hasil disimpan ke: %s", cfg["output_json"])

    return output

# ---------------------------------------------------------------------------
# Pretty-print tabel hasil
# ---------------------------------------------------------------------------

def print_results(output: dict):
    k = output["config"]["top_k"]
    methods = output["methods"]

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "═" * 90)
    print(f"  HASIL EVALUASI RETRIEVAL  |  top_k={k}")
    print("═" * 90)

    col_w = 14
    header = f"{'Metrik':<22}" + "".join(f"{m:>{col_w}}" for m in methods)
    print(header)
    print("─" * 90)

    metric_labels = {
        f"mean_precision_at_k": f"Precision@{k} (mean)",
        f"mean_recall_at_k":    f"Recall@{k}    (mean)",
        "mean_mrr":             "MRR           (mean)",
        f"mean_ndcg_at_k":      f"nDCG@{k}      (mean)",
        "map":                  "MAP           (mean)",
    }

    for key, label in metric_labels.items():
        row = f"  {label:<20}"
        best_val = max(
            output["results"][m]["aggregate"].get(key, 0) for m in methods
        )
        for m in methods:
            val = output["results"][m]["aggregate"].get(key, 0)
            marker = " ★" if val == best_val else "  "
            row += f"{val:>{col_w - 2}.4f}{marker}"
        print(row)

    print("─" * 90)
    print("  ★ = metode terbaik untuk metrik ini\n")

    # ── Per-query breakdown ───────────────────────────────────────────────
    print("═" * 90)
    print("  BREAKDOWN PER QUERY")
    print("═" * 90)

    for method in methods:
        print(f"\n  [{method}]")
        print(f"  {'QID':<6} {'Query':<45} {'P@K':>6} {'R@K':>6} {'MRR':>6} {'nDCG':>6} {'Hits':>5}")
        print("  " + "─" * 82)
        for qr in output["results"][method]["per_query"]:
            m   = qr["metrics"]
            qid = qr["query_id"]
            q   = qr["query"][:43] + ("…" if len(qr["query"]) > 43 else " ")
            hits_str = f"{m['hits']}/{k}"
            print(
                f"  {qid:<6} {q:<45} "
                f"{m['precision_at_k']:>6.3f} "
                f"{m['recall_at_k']:>6.3f} "
                f"{m['mrr']:>6.3f} "
                f"{m['ndcg_at_k']:>6.3f} "
                f"{hits_str:>5}"
            )

    print("\n")


# ---------------------------------------------------------------------------
# Ringkasan interpretasi
# ---------------------------------------------------------------------------

def print_interpretation(output: dict):
    k = output["config"]["top_k"]
    results = output["results"]
    methods = output["methods"]

    print("═" * 90)
    print("  INTERPRETASI METRIK")
    print("═" * 90)
    print(f"""
  Precision@{k}  : Dari {k} dokumen yang diambil, berapa persen yang benar-benar relevan?
                  → Mengukur kualitas / akurasi hasil retrieval.
                  → Nilai tinggi = sedikit noise dalam hasil.

  Recall@{k}     : Dari semua dokumen relevan yang ada*, berapa persen berhasil ditemukan?
                  → Mengukur kelengkapan hasil retrieval.
                  → (* estimasi: 1 chunk relevan per keyword ground truth)
                  → Nilai tinggi = jarang melewatkan dokumen penting.

  MRR            : Mean Reciprocal Rank — seberapa tinggi dokumen relevan PERTAMA muncul?
                  → MRR=1.0 berarti selalu di posisi #1.
                  → MRR=0.5 berarti rata-rata di posisi #2.
                  → Cocok untuk QA system di mana jawaban pertama paling penting.

  nDCG@{k}       : Normalized Discounted Cumulative Gain — mempertimbangkan POSISI.
                  → Dokumen relevan di posisi #1 bernilai lebih dari posisi #5.
                  → nDCG=1.0 = urutan sempurna.

  MAP            : Mean Average Precision — rata-rata AP semua query.
                  → Metrik paling komprehensif, mempertimbangkan posisi semua dok relevan.
""")

    # Rekomendasi otomatis
    print("  REKOMENDASI:")
    agg = {m: results[m]["aggregate"] for m in methods}
    best_mrr    = max(methods, key=lambda m: agg[m].get("mean_mrr", 0))
    best_recall = max(methods, key=lambda m: agg[m].get(f"mean_recall_at_k", 0))
    best_ndcg   = max(methods, key=lambda m: agg[m].get(f"mean_ndcg_at_k", 0))

    print(f"  → MRR terbaik    : {best_mrr}  — pilih ini jika ingin jawaban pertama selalu tepat (RAG/QA)")
    print(f"  → Recall terbaik : {best_recall}  — pilih ini jika tidak boleh melewatkan dokumen relevan")
    print(f"  → nDCG terbaik   : {best_ndcg}  — pilih ini jika urutan ranking penting\n")
    print("═" * 90)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    output = run_evaluation(CONFIG)
    print_results(output)
    print_interpretation(output)
    print(f"  File detail: {CONFIG['output_json']}")
    print("  Buka eval_viewer.html untuk visualisasi lengkap.\n")
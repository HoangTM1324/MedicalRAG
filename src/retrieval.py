import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import util

class HybridRetriever:
    def __init__(self, vector_db, embedding_model):
        self.vector_db = vector_db
        self.embedding_model = embedding_model
        
        # Load tất cả docs để dựng BM25
        print("Initializing Hybrid Retriever (BM25 + Dense)...")
        data = vector_db.get(include=['documents', 'embeddings', 'metadatas'])
        self.documents = data['documents']
        self.metadatas = data['metadatas']
        self.doc_embeddings = np.array(data['embeddings'], dtype=np.float32)
        
        # Build BM25 index
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _normalize_scores(self, scores):
        if len(scores) == 0: return []
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s: return [1.0] * len(scores)
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def search(self, query, alpha=0.5, top_k=2):
        # 1. Dense Search
        q_vec = self.embedding_model.embed_query(query)
        dense_scores = util.cos_sim(np.array(q_vec, dtype=np.float32), self.doc_embeddings)[0].tolist()
        
        # 2. Sparse Search (BM25)
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        norm_sparse = self._normalize_scores(sparse_scores)
        
        # 3. Fusion
        combined = []
        for i in range(len(self.documents)):
            final_score = (alpha * dense_scores[i]) + ((1 - alpha) * norm_sparse[i])
            combined.append({
                "text": self.documents[i],
                "metadata": self.metadatas[i],
                "score": final_score
            })
            
        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]

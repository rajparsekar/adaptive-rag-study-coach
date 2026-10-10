import faiss
import numpy as np


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return index


def search_faiss(
    index,
    model,
    chunks,
    query,
    top_k=3
):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    top_k = min(top_k, len(chunks))

    scores, indices = index.search(
        query_embedding.astype("float32"),
        top_k
    )

    results = []

    for i in range(top_k):
        chunk_index = indices[0][i]

        results.append({
            "chunk_id": chunks[chunk_index]["chunk_id"],
            "source": chunks[chunk_index]["source"],
            "page": chunks[chunk_index]["page"],
            "text": chunks[chunk_index]["text"],
            "score": float(scores[0][i])
        })

    return results
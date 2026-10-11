import numpy as np


def _fetch_all(query_factory, batch_size=500):
    """Fetch all matching database rows in batches."""
    rows = []
    offset = 0

    while True:
        response = (
            query_factory()
            .range(offset, offset + batch_size - 1)
            .execute()
        )

        batch = response.data or []
        rows.extend(batch)

        if len(batch) < batch_size:
            break

        offset += batch_size

    return rows


def _format_vector(embedding):
    """Convert an embedding into pgvector's expected string format."""
    values = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if values.size != 384:
        raise ValueError(
            f"Expected a 384-dimensional embedding, got {values.size}."
        )

    return "[" + ",".join(format(float(v), ".9g") for v in values) + "]"


def _parse_vector(value):
    """Convert a stored pgvector value back into a NumPy array."""
    if isinstance(value, (list, tuple, np.ndarray)):
        vector = np.asarray(value, dtype=np.float32)
    else:
        text = str(value).strip()

        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]

        vector = np.fromstring(text, sep=",", dtype=np.float32)

    vector = vector.reshape(-1)

    if vector.size != 384:
        raise ValueError(
            f"Stored embedding has {vector.size} dimensions instead of 384."
        )

    return vector


def _string_list(value):
    """Ensure a database text-array field receives a list of strings."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value] if value.strip() else []

    return [str(item) for item in value if str(item).strip()]


def save_document_chunks(supabase, user_id, chunks, embeddings):
    """
    Save document records, text chunks, and their embeddings.

    Reprocessing a file with the same name for the same user replaces
    its previous stored version.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            "The number of chunks does not match the number of embeddings."
        )

    if embeddings.shape[1] != 384:
        raise ValueError("Expected embeddings with 384 dimensions.")

    if not chunks:
        return

    grouped = {}

    for global_index, chunk in enumerate(chunks):
        filename = str(chunk.get("source", "")).strip()
        content = str(chunk.get("text", "")).strip()

        if not filename or not content:
            raise ValueError(
                "Every chunk must contain a filename and text."
            )

        grouped.setdefault(filename, []).append(
            (global_index, chunk)
        )

    for filename, file_chunks in grouped.items():
        # Remove the previous version of this user's document.
        existing = (
            supabase.table("documents")
            .select("id")
            .eq("user_id", user_id)
            .eq("filename", filename)
            .execute()
        )

        for document in existing.data or []:
            (
                supabase.table("documents")
                .delete()
                .eq("id", document["id"])
                .eq("user_id", user_id)
                .execute()
            )

        # Create the document record.
        result = (
            supabase.table("documents")
            .insert({
                "user_id": user_id,
                "filename": filename,
            })
            .execute()
        )

        if not result.data:
            raise RuntimeError(
                f"Supabase did not return a document ID for {filename}."
            )

        document_id = result.data[0]["id"]

        # Save chunks and their vectors in batches.
        records = []

        for local_index, (global_index, chunk) in enumerate(file_chunks):
            try:
                page_number = int(chunk["page"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Invalid page number in {filename}, chunk {local_index}."
                )

            records.append({
                "document_id": document_id,
                "user_id": user_id,
                "chunk_index": local_index,
                "page_number": page_number,
                "content": chunk["text"],
                "embedding": _format_vector(
                    embeddings[global_index]
                ),
            })

        for start in range(0, len(records), 100):
            (
                supabase.table("document_chunks")
                .insert(records[start:start + 100])
                .execute()
            )


def load_saved_documents(supabase, user_id):
    """
    Load the user's documents, chunks, and embeddings from Supabase.

    Returns:
        chunks, embeddings, filenames
    """
    documents = _fetch_all(
        lambda: (
            supabase.table("documents")
            .select("id, filename")
            .eq("user_id", user_id)
            .order("created_at")
        )
    )

    if not documents:
        return [], np.empty((0, 384), dtype=np.float32), []

    filename_by_id = {
        document["id"]: document["filename"]
        for document in documents
    }

    rows = _fetch_all(
        lambda: (
            supabase.table("document_chunks")
            .select(
                "document_id, chunk_index, page_number, "
                "content, embedding"
            )
            .eq("user_id", user_id)
            .order("document_id")
            .order("chunk_index")
        )
    )

    loaded_chunks = []
    loaded_embeddings = []

    for row in rows:
        document_id = row["document_id"]

        if document_id not in filename_by_id:
            continue

        loaded_chunks.append({
            "source": filename_by_id[document_id],
            "page": row["page_number"],
            "text": row["content"],
        })

        loaded_embeddings.append(
            _parse_vector(row["embedding"])
        )

    if loaded_embeddings:
        embedding_array = np.vstack(
            loaded_embeddings
        ).astype(np.float32)
    else:
        embedding_array = np.empty(
            (0, 384), dtype=np.float32
        )

    filenames = list(dict.fromkeys(
        document["filename"] for document in documents
    ))

    return loaded_chunks, embedding_array, filenames


def save_evaluation(supabase, user_id, question, student_answer, evaluation):
    """Persist one successfully scored student answer."""
    score = evaluation.get("score")

    if score is None:
        return

    score = int(score)

    if not 0 <= score <= 10:
        raise ValueError("Evaluation score must be between 0 and 10.")

    criteria_scores = evaluation.get("criteria_scores", {})
    if not isinstance(criteria_scores, dict):
        criteria_scores = {}

    record = {
        "user_id": user_id,
        "question": question,
        "student_answer": student_answer,
        "score": score,
        "criteria_scores": criteria_scores,
        "correct_points": _string_list(
            evaluation.get("correct_points")
        ),
        "missing_points": _string_list(
            evaluation.get("missing_points")
        ),
        "weak_concepts": _string_list(
            evaluation.get("weak_concepts")
        ),
        "feedback": str(evaluation.get("feedback") or ""),
        "followup_question": str(
            evaluation.get("followup_question")
            or evaluation.get("follow_up_question")
            or ""
        ),
    }

    (
        supabase.table("answer_evaluations")
        .insert(record)
        .execute()
    )


def load_evaluation_history(supabase, user_id):
    """Load previous evaluations belonging to the logged-in user."""
    rows = _fetch_all(
        lambda: (
            supabase.table("answer_evaluations")
            .select(
                "question, student_answer, score, "
                "criteria_scores, correct_points, missing_points, "
                "weak_concepts, feedback, followup_question, created_at"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
        )
    )

    history = []

    for row in rows:
        history.append({
            "question": row.get("question", ""),
            "answer": row.get("student_answer", ""),
            "score": row.get("score", 0),
            "weak_concepts": row.get("weak_concepts") or [],
            "missing_points": row.get("missing_points") or [],
            "correct_points": row.get("correct_points") or [],
            "criteria_scores": row.get("criteria_scores") or {},
            "feedback": row.get("feedback") or "",
            "followup_question": row.get("followup_question") or "",
            "created_at": row.get("created_at", ""),
        })

    return history
import streamlit as st

from src.pdf_processor import extract_text_from_pdf, split_text
from src.embeddings import load_embedding_model, create_embeddings
from src.retriever import create_faiss_index, search_faiss
from src.gemini_client import (
    load_gemini_client,
    generate_answer,
    evaluate_student_answer,
    generate_adaptive_question,
)


st.set_page_config(
    page_title="Adaptive RAG Study Coach",
    page_icon="📚",
    layout="wide"
)

st.title("Adaptive RAG Study Coach")
st.write(
    "Upload study notes, ask questions, check your answers, "
    "and practise topics that need improvement."
)


@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_gemini_client():
    return load_gemini_client(
        st.secrets["GEMINI_API_KEY"]
    )


# Initialize session data
defaults = {
    "chunks": [],
    "index": None,
    "processed_files": [],
    "current_question": "",
    "retrieved_chunks": [],
    "current_answer": "",
    "evaluation": None,
    "evaluation_history": [],
    "adaptive_question": "",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# 1. UPLOAD AND PROCESS PDFs
# --------------------------------------------------

st.header("1. Study Materials")

uploaded_files = st.file_uploader(
    "Upload one or more academic PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process / Re-index Documents", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF.")

    else:
        all_chunks = []

        try:
            with st.spinner("Reading and indexing documents..."):

                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0)

                    pages = extract_text_from_pdf(uploaded_file)

                    file_chunks = split_text(
                        pages,
                        uploaded_file.name
                    )

                    all_chunks.extend(file_chunks)

                if not all_chunks:
                    st.error(
                        "No readable text was found in the PDFs."
                    )

                else:
                    model = get_embedding_model()

                    embeddings = create_embeddings(
                        model,
                        all_chunks
                    )

                    index = create_faiss_index(embeddings)

                    st.session_state["chunks"] = all_chunks
                    st.session_state["index"] = index
                    st.session_state["processed_files"] = [
                        file.name for file in uploaded_files
                    ]

                    # Reset the current practice session
                    st.session_state["current_question"] = ""
                    st.session_state["retrieved_chunks"] = []
                    st.session_state["current_answer"] = ""
                    st.session_state["evaluation"] = None
                    st.session_state["evaluation_history"] = []
                    st.session_state["adaptive_question"] = ""

                    st.success(
                        f"Processed {len(uploaded_files)} PDF(s). "
                        f"Created {len(all_chunks)} chunks."
                    )

        except Exception as error:
            st.error(f"Document processing failed: {error}")


if st.session_state["index"] is not None:

    with st.expander("View Indexed Documents"):
        for filename in st.session_state["processed_files"]:
            st.write(f"• {filename}")


# --------------------------------------------------
# 2. ASK A QUESTION
# --------------------------------------------------

if st.session_state["index"] is not None:

    st.divider()
    st.header("2. Ask Your Study Material")

    with st.form("question_form"):

        question_input = st.text_input(
            "Enter your question:"
        )

        ask_button = st.form_submit_button("Ask Question")

    if ask_button:

        if not question_input.strip():
            st.warning("Please enter a question.")

        else:
            try:
                model = get_embedding_model()

                with st.spinner("Finding relevant study material..."):

                    retrieved_chunks = search_faiss(
                        index=st.session_state["index"],
                        model=model,
                        chunks=st.session_state["chunks"],
                        query=question_input.strip(),
                        top_k=3
                    )

                with st.spinner("Generating answer..."):

                    answer = generate_answer(
                        get_gemini_client(),
                        question_input.strip(),
                        retrieved_chunks
                    )

                st.session_state["current_question"] = (
                    question_input.strip()
                )
                st.session_state["retrieved_chunks"] = retrieved_chunks
                st.session_state["current_answer"] = answer
                st.session_state["evaluation"] = None
                st.session_state["adaptive_question"] = ""

            except Exception as error:
                st.error(f"Could not generate answer: {error}")


# --------------------------------------------------
# 3. DISPLAY ANSWER AND SOURCES
# --------------------------------------------------

if st.session_state["current_answer"]:

    st.divider()
    st.header("3. AI Answer")

    st.write("**Question:**")
    st.write(st.session_state["current_question"])

    st.write(st.session_state["current_answer"])

    st.subheader("Sources Used")

    for i, chunk in enumerate(
        st.session_state["retrieved_chunks"]
    ):

        with st.expander(
            f"Source {i + 1}: {chunk['source']} "
            f"— Page {chunk['page']}"
        ):

            st.write(chunk["text"])

            st.caption(
                f"Similarity score: {chunk['score']:.3f}"
            )


# --------------------------------------------------
# 4. EVALUATE STUDENT ANSWER
# --------------------------------------------------

if st.session_state["current_answer"]:

    st.divider()
    st.header("4. Check Your Answer")

    st.write(
        "Write your own answer to the question above. "
        "The system will compare it with your study material."
    )

    question_key = str(
        abs(hash(st.session_state["current_question"]))
    )

    with st.form("evaluation_form"):

        student_answer = st.text_area(
            "Your answer:",
            height=150,
            key=f"student_answer_{question_key}"
        )

        evaluate_button = st.form_submit_button(
            "Evaluate My Answer"
        )

    if evaluate_button:

        if not student_answer.strip():
            st.warning("Please write an answer first.")

        else:
            try:
                with st.spinner("Checking your answer..."):

                    evaluation = evaluate_student_answer(
                        get_gemini_client(),
                        st.session_state["current_question"],
                        student_answer.strip(),
                        st.session_state["retrieved_chunks"]
                    )

                st.session_state["evaluation"] = evaluation

                # Record a successfully scored answer once
                if evaluation.get("score") is not None:

                    st.session_state["evaluation_history"].append({
                        "question": st.session_state["current_question"],
                        "answer": student_answer.strip(),
                        "score": evaluation["score"],
                        "weak_concepts": evaluation.get(
                            "weak_concepts", []
                        ),
                        "missing_points": evaluation.get(
                            "missing_points", []
                        ),
                        "feedback": evaluation.get("feedback", ""),
                    })

            except Exception as error:
                st.error(f"Evaluation failed: {error}")


# --------------------------------------------------
# 5. DISPLAY RUBRIC SCORES
# --------------------------------------------------

evaluation = st.session_state["evaluation"]

if evaluation is not None:

    st.divider()
    st.header("5. Evaluation Results")

    score = evaluation.get("score")

    if score is not None:
        st.metric("Final Score", f"{score}/10")

        criteria_scores = evaluation.get(
            "criteria_scores", {}
        )

        criteria_reasons = evaluation.get(
            "criteria_reasons", {}
        )

        criteria_info = [
            ("conceptual_correctness", "Concept correctness", 3),
            ("key_point_coverage", "Important points covered", 3),
            ("explanation", "Explanation", 2),
            ("relevance", "Relevance to the question", 1),
            ("evidence_alignment", "Agreement with study material", 1),
        ]

        st.subheader("Where Your Marks Came From")

        for key, label, maximum in criteria_info:

            marks = criteria_scores.get(key, 0)

            st.write(f"**{label}: {marks}/{maximum}**")

            reason = criteria_reasons.get(key, "")

            if reason:
                st.caption(reason)

    else:
        st.warning("A reliable score could not be calculated.")

    st.subheader("Overall Feedback")
    st.write(evaluation.get("feedback", ""))

    st.subheader("What You Got Right")

    correct_points = evaluation.get("correct_points", [])

    if correct_points:
        for point in correct_points:
            st.write(f"✓ {point}")
    else:
        st.write("No specific correct points were identified.")

    st.subheader("What You Missed")

    missing_points = evaluation.get("missing_points", [])

    if missing_points:
        for point in missing_points:
            st.write(f"• {point}")
    else:
        st.write("No missing points were identified.")

    st.subheader("Topics to Revise")

    weak_concepts = evaluation.get("weak_concepts", [])

    if weak_concepts:
        for concept in weak_concepts:
            st.write(f"• {concept}")
    else:
        st.write("No specific weak concepts were identified.")


# --------------------------------------------------
# 6. GENERATE A TARGETED PRACTICE QUESTION
# --------------------------------------------------

if evaluation is not None and evaluation.get("score") is not None:

    st.divider()
    st.header("6. Targeted Practice")

    weak_concepts = evaluation.get("weak_concepts", [])
    missing_points = evaluation.get("missing_points", [])

    if st.button("Generate Practice Question for My Weak Area"):

        # Select a topic that needs improvement
        if weak_concepts:
            target_concept = weak_concepts[0]

        elif missing_points:
            target_concept = missing_points[0]

        else:
            target_concept = st.session_state["current_question"]

        try:
            with st.spinner(
                f"Preparing a question about {target_concept}..."
            ):

                supporting_chunks = search_faiss(
                    index=st.session_state["index"],
                    model=get_embedding_model(),
                    chunks=st.session_state["chunks"],
                    query=target_concept,
                    top_k=3
                )

                new_question = generate_adaptive_question(
                    get_gemini_client(),
                    target_concept,
                    supporting_chunks
                )

                if new_question:
                    st.session_state["adaptive_question"] = new_question
                else:
                    st.warning("No practice question was generated.")

        except Exception as error:
            st.error(f"Could not generate practice question: {error}")


if st.session_state["adaptive_question"]:

    st.info(
        f"**Your targeted practice question:**\n\n"
        f"{st.session_state['adaptive_question']}"
    )

    if st.button("Practice This Question"):

        try:
            model = get_embedding_model()
            new_question = st.session_state["adaptive_question"]

            with st.spinner("Preparing your practice question..."):

                retrieved_chunks = search_faiss(
                    index=st.session_state["index"],
                    model=model,
                    chunks=st.session_state["chunks"],
                    query=new_question,
                    top_k=3
                )

                answer = generate_answer(
                    get_gemini_client(),
                    new_question,
                    retrieved_chunks
                )

                st.session_state["current_question"] = new_question
                st.session_state["retrieved_chunks"] = retrieved_chunks
                st.session_state["current_answer"] = answer
                st.session_state["evaluation"] = None
                st.session_state["adaptive_question"] = ""

                st.rerun()

        except Exception as error:
            st.error(f"Could not start practice: {error}")


# --------------------------------------------------
# 7. LEARNING HISTORY
# --------------------------------------------------

history = st.session_state["evaluation_history"]

if history:

    st.divider()
    st.header("7. Learning History")

    scores = [
        item["score"]
        for item in history
        if isinstance(item.get("score"), (int, float))
    ]

    average_score = sum(scores) / len(scores) if scores else 0

    col1, col2 = st.columns(2)

    col1.metric("Answers Evaluated", len(scores))
    col2.metric("Average Score", f"{average_score:.1f}/10")

    # Count concepts repeatedly flagged for revision
    concept_counts = {}

    for item in history:
        for concept in item.get("weak_concepts", []):

            name = str(concept).strip()

            if name:
                key = name.casefold()

                if key not in concept_counts:
                    concept_counts[key] = {
                        "Concept": name,
                        "Times Flagged": 0
                    }

                concept_counts[key]["Times Flagged"] += 1

    if concept_counts:

        st.subheader("Frequently Flagged Concepts")

        ranked = sorted(
            concept_counts.values(),
            key=lambda item: item["Times Flagged"],
            reverse=True
        )

        st.dataframe(
            ranked,
            hide_index=True,
            use_container_width=True
        )

        st.caption(
            "These counts show how often Gemini flagged a concept. "
            "They are revision indicators, not a proven measure of mastery."
        )

    st.subheader("Previous Answers")

    for item in reversed(history):

        with st.expander(
            f"{item['question']} — {item['score']}/10"
        ):

            st.write("**Your answer:**")
            st.write(item["answer"])

            st.write("**Feedback:**")
            st.write(item["feedback"])

            st.write("**Topics to revise:**")

            for concept in item.get("weak_concepts", []):
                st.write(f"• {concept}")
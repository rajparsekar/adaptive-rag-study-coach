import streamlit as st

from src.supabase_client import get_supabase_client
from src.supabase_storage import (
    save_document_chunks,
    load_saved_documents,
    save_evaluation,
    load_evaluation_history,
)
from src.pdf_processor import extract_text_from_pdf, split_text
from src.embeddings import load_embedding_model, create_embeddings
from src.retriever import create_faiss_index, search_faiss
from src.gemini_client import (
    load_gemini_client,
    generate_answer,
    evaluate_student_answer,
    generate_adaptive_question,
)


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Adaptive RAG Study Coach",
    page_icon="📚",
    layout="wide",
)

APP_URL = "https://adaptive-rag-study-coach.streamlit.app/"


# --------------------------------------------------
# SUPABASE CONNECTION
# --------------------------------------------------

if "supabase_client" not in st.session_state:
    st.session_state["supabase_client"] = get_supabase_client()

if "supabase_user" not in st.session_state:
    st.session_state["supabase_user"] = None

if "password_recovery_verified" not in st.session_state:
    st.session_state["password_recovery_verified"] = False

if "password_recovery_error" not in st.session_state:
    st.session_state["password_recovery_error"] = ""

if "password_reset_success" not in st.session_state:
    st.session_state["password_reset_success"] = False

if "_processed_recovery_token_hash" not in st.session_state:
    st.session_state["_processed_recovery_token_hash"] = ""

supabase = st.session_state["supabase_client"]


# --------------------------------------------------
# VERIFY PASSWORD RECOVERY LINK
# --------------------------------------------------

token_hash = st.query_params.get("token_hash", "")
recovery_type = st.query_params.get("type", "")

if token_hash and recovery_type == "recovery":
    if (
        st.session_state["_processed_recovery_token_hash"]
        != token_hash
    ):
        try:
            with st.spinner("Verifying your password reset link..."):
                response = supabase.auth.verify_otp(
                    {
                        "token_hash": token_hash,
                        "type": "recovery",
                    }
                )

            if response.user is None:
                raise ValueError(
                    "Supabase did not return a verified user."
                )

            st.session_state["password_recovery_verified"] = True
            st.session_state["password_recovery_error"] = ""

        except Exception:
            st.session_state["password_recovery_verified"] = False
            st.session_state["password_recovery_error"] = (
                "This password reset link is invalid or has expired. "
                "Please request a new password reset email."
            )

        finally:
            # Avoid trying to verify the same one-time token on every rerun.
            st.session_state["_processed_recovery_token_hash"] = (
                token_hash
            )


# --------------------------------------------------
# SET A NEW PASSWORD
# --------------------------------------------------

if st.session_state["password_recovery_verified"]:
    st.title("Reset Your Password")
    st.write(
        "Your recovery link has been verified. "
        "Enter and confirm your new password below."
    )

    with st.form("new_password_form"):
        new_password = st.text_input(
            "New password",
            type="password",
            key="new_password",
        )

        confirm_password = st.text_input(
            "Confirm new password",
            type="password",
            key="confirm_new_password",
        )

        update_password_button = st.form_submit_button(
            "Update Password",
            type="primary",
        )

    if update_password_button:
        if not new_password or not confirm_password:
            st.warning("Enter and confirm your new password.")

        elif len(new_password) < 8:
            st.warning(
                "Use a password with at least 8 characters."
            )

        elif new_password != confirm_password:
            st.error("The passwords do not match.")

        else:
            try:
                with st.spinner("Updating your password..."):
                    supabase.auth.update_user(
                        {"password": new_password}
                    )

                # End the recovery session so the user logs in again.
                try:
                    supabase.auth.sign_out()
                except Exception:
                    pass

                st.session_state["supabase_user"] = None
                st.session_state["password_recovery_verified"] = False
                st.session_state["password_recovery_error"] = ""
                st.session_state["password_reset_success"] = True
                st.session_state["_processed_recovery_token_hash"] = ""

                # Clear the one-time token from the browser URL.
                st.query_params.clear()

                # Use a clean client for the normal login screen.
                st.session_state["supabase_client"] = (
                    get_supabase_client()
                )

                # Clear user-specific in-memory information.
                for key in (
                    "chunks",
                    "index",
                    "processed_files",
                    "current_question",
                    "retrieved_chunks",
                    "current_answer",
                    "evaluation",
                    "evaluation_history",
                    "adaptive_question",
                    "_storage_loaded_user_id",
                ):
                    st.session_state.pop(key, None)

                st.rerun()

            except Exception as error:
                st.error(
                    "Could not update the password. "
                    f"Please try again. Details: {error}"
                )

    st.stop()


# --------------------------------------------------
# LOGIN, SIGN-UP AND FORGOT PASSWORD
# --------------------------------------------------

if st.session_state["supabase_user"] is None:
    st.title("Adaptive RAG Study Coach")
    st.write("Log in or create an account to continue.")

    if st.session_state["password_reset_success"]:
        st.success(
            "Your password has been updated successfully. "
            "Log in using your new password."
        )
        st.session_state["password_reset_success"] = False

    if st.session_state["password_recovery_error"]:
        st.warning(
            st.session_state["password_recovery_error"]
        )

    login_tab, signup_tab, forgot_tab = st.tabs(
        ["Log In", "Create Account", "Forgot Password"]
    )

    # --------------------------------------------------
    # LOG IN
    # --------------------------------------------------

    with login_tab:
        with st.form("login_form"):
            login_email = st.text_input(
                "Email",
                key="login_email",
            )

            login_password = st.text_input(
                "Password",
                type="password",
                key="login_password",
            )

            login_button = st.form_submit_button("Log In")

        if login_button:
            if not login_email.strip() or not login_password:
                st.warning("Enter your email and password.")

            else:
                try:
                    response = supabase.auth.sign_in_with_password(
                        {
                            "email": login_email.strip(),
                            "password": login_password,
                        }
                    )

                    st.session_state["supabase_user"] = {
                        "id": response.user.id,
                        "email": (
                            response.user.email
                            or login_email.strip()
                        ),
                    }

                    st.session_state["password_recovery_error"] = ""
                    st.rerun()

                except Exception as error:
                    st.error(f"Login failed: {error}")

    # --------------------------------------------------
    # CREATE ACCOUNT
    # --------------------------------------------------

    with signup_tab:
        with st.form("signup_form"):
            signup_email = st.text_input(
                "Email",
                key="signup_email",
            )

            signup_password = st.text_input(
                "Password",
                type="password",
                key="signup_password",
            )

            signup_button = st.form_submit_button(
                "Create Account"
            )

        if signup_button:
            if not signup_email.strip() or not signup_password:
                st.warning("Enter your email and password.")

            elif len(signup_password) < 8:
                st.warning(
                    "Use a password with at least 8 characters."
                )

            else:
                try:
                    response = supabase.auth.sign_up(
                        {
                            "email": signup_email.strip(),
                            "password": signup_password,
                        }
                    )

                    if (
                        response.session is not None
                        and response.user is not None
                    ):
                        st.session_state["supabase_user"] = {
                            "id": response.user.id,
                            "email": (
                                response.user.email
                                or signup_email.strip()
                            ),
                        }
                        st.rerun()

                    else:
                        st.success(
                            "Account created. Check your email for "
                            "the confirmation link, then log in."
                        )

                except Exception as error:
                    st.error(f"Sign-up failed: {error}")

    # --------------------------------------------------
    # FORGOT PASSWORD
    # --------------------------------------------------

    with forgot_tab:
        st.write(
            "Enter the email associated with your account. "
            "We'll send you a password reset link."
        )

        with st.form("forgot_password_form"):
            reset_email = st.text_input(
                "Account email",
                key="reset_email",
            )

            reset_button = st.form_submit_button(
                "Send Reset Link"
            )

        if reset_button:
            if not reset_email.strip():
                st.warning("Enter your email address.")

            else:
                try:
                    st.session_state["password_recovery_error"] = ""

                    with st.spinner("Requesting reset email..."):
                        supabase.auth.reset_password_for_email(
                            reset_email.strip(),
                            {
                                "redirect_to": APP_URL,
                            },
                        )

                    st.success(
                        "If an account exists for that address, "
                        "a password reset email has been requested. "
                        "Check your inbox and spam folder."
                    )

                except Exception as error:
                    st.error(
                        "Could not request a reset email. "
                        f"Please try again. Details: {error}"
                    )

    st.stop()


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

user = st.session_state["supabase_user"]
user_id = user["id"]

st.sidebar.caption(f"Logged in as: {user['email']}")

if st.sidebar.button("Log Out"):
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    st.session_state["supabase_user"] = None

    for key in (
        "chunks",
        "index",
        "processed_files",
        "current_question",
        "retrieved_chunks",
        "current_answer",
        "evaluation",
        "evaluation_history",
        "adaptive_question",
        "_storage_loaded_user_id",
    ):
        st.session_state.pop(key, None)

    st.session_state["supabase_client"] = get_supabase_client()
    st.rerun()


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return load_embedding_model()


@st.cache_resource
def get_gemini_client():
    return load_gemini_client(
        st.secrets["GEMINI_API_KEY"]
    )


# --------------------------------------------------
# INITIALIZE SESSION DATA
# --------------------------------------------------

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
# LOAD SAVED DOCUMENTS AND LEARNING HISTORY
# --------------------------------------------------

if st.session_state.get("_storage_loaded_user_id") != user_id:
    try:
        with st.spinner("Loading your saved study data..."):
            saved_chunks, saved_embeddings, saved_files = (
                load_saved_documents(supabase, user_id)
            )

            if (
                len(saved_chunks) > 0
                and saved_embeddings.shape[0] > 0
            ):
                saved_index = create_faiss_index(saved_embeddings)
            else:
                saved_index = None

            saved_history = load_evaluation_history(
                supabase, user_id
            )

        st.session_state["chunks"] = saved_chunks
        st.session_state["index"] = saved_index
        st.session_state["processed_files"] = saved_files
        st.session_state["evaluation_history"] = saved_history

        st.session_state["current_question"] = ""
        st.session_state["retrieved_chunks"] = []
        st.session_state["current_answer"] = ""
        st.session_state["evaluation"] = None
        st.session_state["adaptive_question"] = ""
        st.session_state["_storage_loaded_user_id"] = user_id

    except Exception as error:
        st.error(
            f"Could not load your saved Supabase data: {error}"
        )
        st.stop()


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

st.title("Adaptive RAG Study Coach")
st.write(
    "Upload study notes, ask questions, check your answers, "
    "and practise topics that need improvement."
)


# --------------------------------------------------
# 1. UPLOAD AND PROCESS PDFs
# --------------------------------------------------

st.header("1. Study Materials")

uploaded_files = st.file_uploader(
    "Upload one or more academic PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button(
    "Process / Re-index Documents",
    type="primary",
):
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
                        uploaded_file.name,
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
                        all_chunks,
                    )
                    index = create_faiss_index(embeddings)

                    with st.spinner(
                        "Saving study materials to Supabase..."
                    ):
                        save_document_chunks(
                            supabase,
                            user_id,
                            all_chunks,
                            embeddings,
                        )

                        current_filenames = {
                            file.name for file in uploaded_files
                        }

                        existing_response = (
                            supabase.table("documents")
                            .select("id, filename")
                            .eq("user_id", user_id)
                            .execute()
                        )

                        for document in existing_response.data or []:
                            if (
                                document["filename"]
                                not in current_filenames
                            ):
                                (
                                    supabase.table("documents")
                                    .delete()
                                    .eq("user_id", user_id)
                                    .eq("id", document["id"])
                                    .execute()
                                )

                    st.session_state["chunks"] = all_chunks
                    st.session_state["index"] = index
                    st.session_state["processed_files"] = sorted(
                        current_filenames
                    )

                    st.session_state["current_question"] = ""
                    st.session_state["retrieved_chunks"] = []
                    st.session_state["current_answer"] = ""
                    st.session_state["evaluation"] = None
                    st.session_state["adaptive_question"] = ""

                    st.session_state["evaluation_history"] = (
                        load_evaluation_history(supabase, user_id)
                    )

                    st.success(
                        f"Processed {len(uploaded_files)} PDF(s). "
                        f"Created {len(all_chunks)} chunks and "
                        "saved them to Supabase."
                    )

        except Exception as error:
            st.error(
                f"Document processing or saving failed: {error}"
            )


if st.session_state["index"] is not None:
    with st.expander("View Indexed Documents"):
        for filename in st.session_state["processed_files"]:
            st.write(f"• {filename}")
else:
    st.info(
        "Upload and process PDFs to start asking questions."
    )


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

                with st.spinner(
                    "Finding relevant study material..."
                ):
                    retrieved_chunks = search_faiss(
                        index=st.session_state["index"],
                        model=model,
                        chunks=st.session_state["chunks"],
                        query=question_input.strip(),
                        top_k=3,
                    )

                with st.spinner("Generating answer..."):
                    answer = generate_answer(
                        get_gemini_client(),
                        question_input.strip(),
                        retrieved_chunks,
                    )

                st.session_state["current_question"] = (
                    question_input.strip()
                )
                st.session_state["retrieved_chunks"] = (
                    retrieved_chunks
                )
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

            if "score" in chunk:
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
            key=f"student_answer_{question_key}",
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
                    evaluation_result = evaluate_student_answer(
                        get_gemini_client(),
                        st.session_state["current_question"],
                        student_answer.strip(),
                        st.session_state["retrieved_chunks"],
                    )

                st.session_state["evaluation"] = evaluation_result

                if evaluation_result.get("score") is not None:
                    try:
                        save_evaluation(
                            supabase=supabase,
                            user_id=user_id,
                            question=st.session_state[
                                "current_question"
                            ],
                            student_answer=student_answer.strip(),
                            evaluation=evaluation_result,
                        )

                        st.success(
                            "Your evaluation has been saved "
                            "to your account."
                        )

                    except Exception as save_error:
                        st.warning(
                            "Your answer was evaluated, but Supabase "
                            f"could not save it: {save_error}"
                        )

                    st.session_state["evaluation_history"].insert(
                        0,
                        {
                            "question": st.session_state[
                                "current_question"
                            ],
                            "answer": student_answer.strip(),
                            "score": evaluation_result["score"],
                            "weak_concepts": evaluation_result.get(
                                "weak_concepts", []
                            ),
                            "missing_points": evaluation_result.get(
                                "missing_points", []
                            ),
                            "correct_points": evaluation_result.get(
                                "correct_points", []
                            ),
                            "criteria_scores": evaluation_result.get(
                                "criteria_scores", {}
                            ),
                            "feedback": evaluation_result.get(
                                "feedback", ""
                            ),
                        },
                    )

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
            (
                "conceptual_correctness",
                "Concept correctness",
                3,
            ),
            (
                "key_point_coverage",
                "Important points covered",
                3,
            ),
            ("explanation", "Explanation", 2),
            ("relevance", "Relevance to the question", 1),
            (
                "evidence_alignment",
                "Agreement with study material",
                1,
            ),
        ]

        st.subheader("Where Your Marks Came From")

        for key, label, maximum in criteria_info:
            marks = criteria_scores.get(key, 0)
            st.write(f"**{label}: {marks}/{maximum}**")

            reason = criteria_reasons.get(key, "")
            if reason:
                st.caption(reason)

    else:
        st.warning(
            "A reliable score could not be calculated."
        )

    st.subheader("Overall Feedback")
    st.write(evaluation.get("feedback", ""))

    st.subheader("What You Got Right")
    correct_points = evaluation.get("correct_points", [])

    if correct_points:
        for point in correct_points:
            st.write(f"✓ {point}")
    else:
        st.write(
            "No specific correct points were identified."
        )

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

if (
    evaluation is not None
    and evaluation.get("score") is not None
):
    st.divider()
    st.header("6. Targeted Practice")

    weak_concepts = evaluation.get("weak_concepts", [])
    missing_points = evaluation.get("missing_points", [])

    if st.button(
        "Generate Practice Question for My Weak Area"
    ):
        if weak_concepts:
            target_concept = weak_concepts[0]
        elif missing_points:
            target_concept = missing_points[0]
        else:
            target_concept = st.session_state[
                "current_question"
            ]

        try:
            with st.spinner(
                f"Preparing a question about {target_concept}..."
            ):
                supporting_chunks = search_faiss(
                    index=st.session_state["index"],
                    model=get_embedding_model(),
                    chunks=st.session_state["chunks"],
                    query=target_concept,
                    top_k=3,
                )

                new_question = generate_adaptive_question(
                    get_gemini_client(),
                    target_concept,
                    supporting_chunks,
                )

                if new_question:
                    st.session_state["adaptive_question"] = (
                        new_question
                    )
                else:
                    st.warning(
                        "No practice question was generated."
                    )

        except Exception as error:
            st.error(
                f"Could not generate practice question: {error}"
            )


if st.session_state["adaptive_question"]:
    st.info(
        "**Your targeted practice question:**\n\n"
        f"{st.session_state['adaptive_question']}"
    )

    if st.button("Practice This Question"):
        try:
            model = get_embedding_model()
            new_question = st.session_state["adaptive_question"]

            with st.spinner(
                "Preparing your practice question..."
            ):
                retrieved_chunks = search_faiss(
                    index=st.session_state["index"],
                    model=model,
                    chunks=st.session_state["chunks"],
                    query=new_question,
                    top_k=3,
                )

                answer = generate_answer(
                    get_gemini_client(),
                    new_question,
                    retrieved_chunks,
                )

                st.session_state["current_question"] = (
                    new_question
                )
                st.session_state["retrieved_chunks"] = (
                    retrieved_chunks
                )
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

st.divider()
st.header("7. Learning History")

if history:
    scores = [
        item["score"]
        for item in history
        if isinstance(item.get("score"), (int, float))
    ]

    average_score = (
        sum(scores) / len(scores) if scores else 0
    )

    col1, col2 = st.columns(2)
    col1.metric("Answers Evaluated", len(scores))
    col2.metric(
        "Average Score",
        f"{average_score:.1f}/10",
    )

    concept_counts = {}

    for item in history:
        for concept in item.get("weak_concepts", []):
            name = str(concept).strip()

            if name:
                key = name.casefold()

                if key not in concept_counts:
                    concept_counts[key] = {
                        "Concept": name,
                        "Times Flagged": 0,
                    }

                concept_counts[key]["Times Flagged"] += 1

    if concept_counts:
        st.subheader("Frequently Flagged Concepts")

        ranked = sorted(
            concept_counts.values(),
            key=lambda item: item["Times Flagged"],
            reverse=True,
        )

        st.dataframe(
            ranked,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These counts show how often Gemini flagged a concept. "
            "They are revision indicators, not a proven measure "
            "of mastery."
        )

    st.subheader("Previous Answers")

    for item in history:
        with st.expander(
            f"{item['question']} — {item['score']}/10"
        ):
            st.write("**Your answer:**")
            st.write(item["answer"])

            st.write("**Feedback:**")
            st.write(item.get("feedback", ""))

            st.write("**Topics to revise:**")
            for concept in item.get("weak_concepts", []):
                st.write(f"• {concept}")

else:
    st.info(
        "No evaluated answers saved yet. Evaluate an answer "
        "to build your learning history."
    )
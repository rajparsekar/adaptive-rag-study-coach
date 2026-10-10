# Adaptive RAG Study Coach

A Generative AI study assistant that answers questions from
academic PDFs and helps students identify topics they need
to revise.

## Features

- Upload multiple academic PDF documents.
- Extract and organize text from uploaded files.
- Convert text into embeddings for meaning-based search.
- Retrieve relevant information using FAISS.
- Generate answers using Google Gemini.
- Display source documents and page numbers.
- Evaluate student answers using a marking rubric.
- Identify missing points and topics for revision.
- Generate practice questions targeting weak topics.
- Track scores and weak concepts during the current session.

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- NumPy
- Google Gemini API

## How It Works

1. The user uploads academic PDF files.
2. The application extracts and divides the text into chunks.
3. Sentence Transformers converts the chunks into embeddings.
4. FAISS finds the chunks relevant to the user's question.
5. Gemini generates an answer using the retrieved information.
6. The application displays the answer and its sources.
7. Students can submit their own answers for evaluation and
   receive feedback and targeted practice questions.

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required packages:

   pip install -r requirements.txt

4. Create `.streamlit/secrets.toml` and add your Gemini API key:

   GEMINI_API_KEY = "your_api_key_here"

5. Start the application:

   streamlit run app.py

## Security

Never commit API keys or `.streamlit/secrets.toml` to GitHub.

## Current Limitations

Learning history is currently stored in the active Streamlit
session. Persistent storage across sessions is planned.

## Future Improvements

- Persistent student progress and learning history.
- Improved document retrieval and answer evaluation.
- More adaptive practice features.
- Online deployment and mobile-browser testing.
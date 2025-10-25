# app.py
import os
import io
import hashlib
import threading
from datetime import datetime
import tempfile
from typing import List, Tuple

import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import google.generativeai as genai
from googletrans import Translator

# Voice I/O
import speech_recognition as sr
import pyttsx3

# LangChain (0.2+)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

# NEW: dotenv for environment variables
from dotenv import load_dotenv

# ------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------

# Load .env first; this won't error if file is absent
load_dotenv()

def _get_google_api_key() -> str | None:
    # Priority 1: .env / process env
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key

    # Priority 2: Streamlit secrets, but only if available (avoid raising)
    try:
        key = st.secrets["GOOGLE_API_KEY"]
        if key:
            return key
    except Exception:
        pass

    return None

GOOGLE_API_KEY = _get_google_api_key()
if not GOOGLE_API_KEY:
    st.error(
        "Missing GOOGLE_API_KEY. Create a `.env` file in your project folder with:\n\n"
        "GOOGLE_API_KEY=your_real_api_key_here\n\n"
        "Then restart the app."
    )
    st.stop()

# Configure the Google Generative AI SDK
genai.configure(api_key=GOOGLE_API_KEY)

# Vector store persistence path
FAISS_INDEX_DIR = "faiss_index"

# ------------------------------------------------------------------------------------
# Session & state
# ------------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Track TTS state
if "tts_is_speaking" not in st.session_state:
    st.session_state["tts_is_speaking"] = False
if "last_answer_text" not in st.session_state:
    st.session_state["last_answer_text"] = ""

# Initialize a single global pyttsx3 engine
# (Initialize once so .stop() can affect the active engine)
_engine_lock = threading.Lock()
_engine = None

def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            try:
                _engine = pyttsx3.init()
            except Exception as e:
                st.info(f"TTS engine not available: {e}")
                _engine = None
        return _engine

# ------------------------------------------------------------------------------------
# Audio utilities (local TTS, start/stop only)
# ------------------------------------------------------------------------------------
def start_speaking(text: str):
    """
    Start speaking in a background thread. If already speaking, do nothing.
    """
    if not text:
        return
    if st.session_state.get("tts_is_speaking"):
        return
    engine = _get_engine()
    if engine is None:
        st.warning("TTS engine is not available on this machine.")
        return

    def _speak_worker(t: str):
        try:
            with _engine_lock:
                engine.say(t)
                st.session_state["tts_is_speaking"] = True
            engine.runAndWait()
        except Exception as e:
            st.info(f"TTS failed: {e}")
        finally:
            st.session_state["tts_is_speaking"] = False

    th = threading.Thread(target=_speak_worker, args=(text,), daemon=True)
    th.start()

def stop_speaking():
    """
    Stop speaking immediately.
    """
    engine = _get_engine()
    if engine is None:
        return
    try:
        with _engine_lock:
            engine.stop()
    except Exception:
        pass
    finally:
        st.session_state["tts_is_speaking"] = False

def speech_to_text() -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            st.write("You said:", text)
            return text
    except sr.UnknownValueError:
        st.error("Sorry, I could not understand the audio.")
    except sr.RequestError:
        st.error("Speech recognition service is not available.")
    except Exception as e:
        st.error(f"Speech capture failed: {e}")
    return ""

# ------------------------------------------------------------------------------------
# PDF ingestion
# ------------------------------------------------------------------------------------
def read_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a single PDF (bytes)."""
    text = ""
    with io.BytesIO(pdf_bytes) as bio:
        reader = PdfReader(bio)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text

def extract_images_from_pdf(pdf_bytes: bytes) -> List[bytes]:
    """Extract embedded images from a single PDF (bytes)."""
    images: List[bytes] = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp_name = tmp.name
    try:
        doc = fitz.open(tmp_name)
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    if base_image and "image" in base_image:
                        images.append(base_image["image"])
                except Exception:
                    continue
        doc.close()
    finally:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
    return images

def get_text_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    return splitter.split_text(text)

# ------------------------------------------------------------------------------------
# Vector store & QA chain (LangChain 0.2+)
# ------------------------------------------------------------------------------------
def build_or_update_vectorstore(chunks: List[str]):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    if not chunks:
        raise ValueError("No text chunks to embed.")
    vs = FAISS.from_texts(chunks, embedding=embeddings)
    vs.save_local(FAISS_INDEX_DIR)

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY,
    )
    return FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

def get_qa_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a precise assistant. Answer the question strictly from the provided context.
If the answer is not present in the context, reply exactly: "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:"""
    )
    doc_chain = create_stuff_documents_chain(llm, prompt)
    return doc_chain

def run_qa(user_question: str) -> str:
    vs = load_vectorstore()
    docs = vs.similarity_search(user_question, k=4)
    chain = get_qa_chain()
    return chain.invoke({"context": docs, "question": user_question})

# ------------------------------------------------------------------------------------
# Translation & summaries
# ------------------------------------------------------------------------------------
def translate_text(text: str, dest_language: str) -> str:
    if not text or dest_language == "en":
        return text
    try:
        translator = Translator()
        return translator.translate(text, dest=dest_language).text
    except Exception as e:
        st.info(f"Translation unavailable: {e}")
        return text

def summarize_pdf_text(pdf_text: str, limit: int = 1000) -> str:
    if len(pdf_text) <= limit:
        return pdf_text
    return pdf_text[:limit] + "..."

# ------------------------------------------------------------------------------------
# Chat management
# ------------------------------------------------------------------------------------
def append_chat(q: str, a: str):
    st.session_state["chat_history"].append({"question": q, "answer": a})
    st.session_state["last_answer_text"] = a  # keep the latest for TTS

def clear_conversation():
    st.session_state["chat_history"] = []
    st.session_state["last_answer_text"] = ""
    stop_speaking()

def delete_prompt(idx: int):
    if 0 <= idx < len(st.session_state["chat_history"]):
        del st.session_state["chat_history"][idx]

def download_chat_history():
    if not st.session_state["chat_history"]:
        return
    chat_history_str = "\n\n".join(
        [f"Q: {entry['question']}\nA: {entry['answer']}" for entry in st.session_state["chat_history"]]
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history_{timestamp}.txt"
    st.download_button("Download Chat History", data=chat_history_str.encode(), file_name=filename)

# ------------------------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="DocuBot", page_icon=":scroll:")
    st.title("📚 DocuBot: Multi-PDF Chatbot 🤖 with Voice & Visuals 🎤")
    st.write("Upload PDFs and ask questions in multiple languages. AI responds via text and voice!")

    # ========= LOGIN REMOVED (commented) =========
    # st.sidebar.title("User Authentication")
    # use_signup = st.sidebar.checkbox("New user? Sign up here")
    # if use_signup:
    #     username = st.sidebar.text_input("Enter Username")
    #     password = st.sidebar.text_input("Enter Password", type="password")
    #     if st.sidebar.button("Sign Up"):
    #         ok, msg = sign_up(username, password)
    #         st.sidebar.write(msg)
    # else:
    #     username = st.sidebar.text_input("Enter Username")
    #     password = st.sidebar.text_input("Enter Password", type="password")
    #     authed = False
    #     if st.sidebar.button("Sign In"):
    #         authed, msg = sign_in(username, password)
    #         st.sidebar.write(msg)
    #         if not authed:
    #             st.stop()
    # =============================================

    # Language
    language = st.selectbox("Select Language:", ["en", "es", "fr", "de", "it"])

    # Query input
    col1, col2 = st.columns(2)
    with col1:
        text_question = st.text_input("Ask a question about the uploaded PDFs… ✍️📝")
    with col2:
        if st.button("🎙️ Click to Speak"):
            spoken = speech_to_text()
        else:
            spoken = ""

    user_question = spoken if spoken else text_question

    # Sidebar: PDF upload & processing
    with st.sidebar:
        st.write("---")
        st.title("📁 PDF Files")
        pdf_files = st.file_uploader(
            "Upload your PDF files and click Submit & Process",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Submit & Process"):
            if not pdf_files:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs…"):
                    all_text = ""
                    all_images = []

                    # Read all files into memory once to avoid pointer issues
                    pdf_blobs: List[bytes] = [f.read() for f in pdf_files]

                    # Text extraction
                    for blob in pdf_blobs:
                        all_text += read_pdf_text(blob)

                    # Chunk & build index
                    chunks = get_text_chunks(all_text)
                    if chunks:
                        build_or_update_vectorstore(chunks)
                    else:
                        st.warning("No extractable text found. You can still view images below.")

                    # Image extraction
                    for blob in pdf_blobs:
                        imgs = extract_images_from_pdf(blob)
                        all_images.extend(imgs)

                    # Show images (if any)
                    if all_images:
                        st.write("### Extracted Images")
                        for img in all_images:
                            st.image(img, use_column_width=True)
                    else:
                        st.info("No embedded images were found.")

                    # Show a quick summary of all text
                    if all_text.strip():
                        st.write("### Summarized Content")
                        st.write(summarize_pdf_text(all_text))
                    else:
                        st.info("No text extracted from the uploaded PDFs.")

    # Handle user question
    if user_question:
        try:
            answer = run_qa(user_question)
        except Exception as e:
            st.error(f"Search/answer failed. Did you click 'Submit & Process' after uploading PDFs? Error: {e}")
            answer = ""

        if answer:
            translated = translate_text(answer, language)
            append_chat(user_question, translated)

            st.write(f"**Q**: {user_question}")
            st.write(f"**A**: {translated}")

            # ------- Simple TTS Toggle (Speak/Stop) -------
            st.write("#### 🔊 Audio")
            if not st.session_state["tts_is_speaking"]:
                if st.button("🔊 Speak this answer"):
                    start_speaking(st.session_state["last_answer_text"])
            else:
                if st.button("⏹️ Stop speaking"):
                    stop_speaking()

    # If speaking, show a status pill
    if st.session_state.get("tts_is_speaking"):
        st.info("Speaking… Click 'Stop speaking' to end.")

    # Conversation controls
    if st.button("Clear Conversation"):
        clear_conversation()

    if st.session_state["chat_history"]:
        st.write("### Conversation History")
        for idx, entry in enumerate(st.session_state["chat_history"]):
            st.write(f"Q: {entry['question']}")
            st.write(f"A: {entry['answer']}")
            if st.button(f"Delete Q&A {idx+1}", key=f"delete_{idx}"):
                delete_prompt(idx)

    download_chat_history()


if __name__ == "__main__":
    main()

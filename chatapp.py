import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr
import pyttsx3
import fitz  # PyMuPDF for extracting visuals
import hashlib
from googletrans import Translator
from datetime import datetime
import threading
import tempfile

# Set default API key
google_api_key = "AIzaSyDQ24jcEZaAOEnaIAxYxA2w4rU6vPwpE1s"
genai.configure(api_key=google_api_key)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Store user session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User authentication - Sign Up
users_db = {}

def sign_up(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if username in users_db:
        return False, "Username already exists!"
    users_db[username] = hashed_pw
    return True, "Account created successfully!"

def sign_in(username, password):
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if users_db.get(username) == hashed_pw:
        return True, "Login successful!"
    return False, "Invalid credentials!"

# Text-to-Speech
def speak_text(text):
    def speak():
        engine.say(text)
        engine.runAndWait()
    
    speak_thread = threading.Thread(target=speak)
    speak_thread.start()

# Speech-to-Text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write("You said: ", text)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Speech recognition service is not available.")
    return ""

# Process PDF Text and Extract Images
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

import tempfile

def extract_images_from_pdf(pdf_docs):
    images = []
    for pdf in pdf_docs:
        # Use a temporary file to store the uploaded PDF content
        with tempfile.NamedTemporaryFile(delete=False, mode="wb") as tmp_pdf:
            tmp_pdf.write(pdf.read())  # Write the uploaded PDF content to the temp file
            tmp_pdf.close()  # Close the temp file to ensure it's saved

            try:
                # Open the PDF using PyMuPDF (fitz)
                doc = fitz.open(tmp_pdf.name)  # Open the temp file with fitz
                for page in doc:
                    image_list = page.get_images(full=True)
                    for img in image_list:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        images.append(base_image['image'])
                doc.close()  # Ensure document is closed after processing

            except Exception as e:
                st.error(f"Error processing the PDF: {e}")

    return images


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User interaction
def user_input(user_question, language="en"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    translated_response = translate_text(response["output_text"], language)
    st.session_state["chat_history"].append({"question": user_question, "answer": translated_response})

    for chat in st.session_state["chat_history"]:
        st.write(f"**Q**: {chat['question']}")
        st.write(f"**A**: {chat['answer']}")
    
    speak_text(translated_response)

# Translate text
def translate_text(text, dest_language):
    translator = Translator()
    return translator.translate(text, dest=dest_language).text

# Summarize PDF
def summarize_pdf(pdf_text):
    return pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text

def display_images(images):
    for img in images:
        st.image(img, use_column_width=True)

# Download chat history
def download_chat_history():
    chat_history_str = "\n\n".join([f"Q: {entry['question']}\nA: {entry['answer']}" for entry in st.session_state["chat_history"]])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"chat_history_{timestamp}.txt"
    st.download_button("Download Chat History", data=chat_history_str.encode(), file_name=filename)

# Clear conversation function
def clear_conversation():
    st.session_state["chat_history"] = []

# Delete a specific prompt
def delete_prompt(index):
    if index < len(st.session_state["chat_history"]):
        del st.session_state["chat_history"][index]

# Main app UI
def main():
    st.set_page_config(page_title="DocuBot", page_icon=":scroll:")
    st.title("📚DocuBot: Multi-PDF Chatbot 🤖 with Voice & Visuals 🎤")
    st.write("### Upload PDFs and ask questions in multiple languages. AI responds via text and voice!")

    st.sidebar.title("User Authentication")
    sign_up_mode = st.sidebar.checkbox("New user? Sign up here")

    if sign_up_mode:
        username = st.sidebar.text_input("Enter Username")
        password = st.sidebar.text_input("Enter Password", type="password")
        if st.sidebar.button("Sign Up"):
            success, message = sign_up(username, password)
            st.sidebar.write(message)
    else:
        username = st.sidebar.text_input("Enter Username")
        password = st.sidebar.text_input("Enter Password", type="password")
        if st.sidebar.button("Sign In"):
            success, message = sign_in(username, password)
            st.sidebar.write(message)
            if not success:
                return

    language = st.selectbox("Select Language:", ["en", "es", "fr", "de", "it"])

    if st.button("Click to Speak"):
        user_question = speech_to_text()
    else:
        user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        user_input(user_question, language)

    if st.button("Clear Conversation"):
        clear_conversation()

    if st.session_state["chat_history"]:
        st.write("### Conversation History:")
        for idx, entry in enumerate(st.session_state["chat_history"]):
            st.write(f"Q: {entry['question']}")
            st.write(f"A: {entry['answer']}")
            if st.button(f"Delete Q&A {idx+1}", key=f"delete_{idx}"):
                delete_prompt(idx)

    with st.sidebar:
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                all_text = ""
                for pdf in pdf_docs:
                    raw_text = get_pdf_text([pdf])
                    all_text += raw_text

                text_chunks = get_text_chunks(all_text)
                get_vector_store(text_chunks)

                images = extract_images_from_pdf(pdf_docs)
                if images:
                    st.write("Extracted Images:")
                    display_images(images)

                summarized_text = summarize_pdf(all_text)
                st.write(f"Summarized Content: {summarized_text}")

    download_chat_history()

if __name__ == "__main__":
    main()

import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# Robust import of transformers.pipeline for broader compatibility
try:
    from transformers import pipeline as hf_pipeline  # standard path
except Exception:
    try:
        from transformers.pipelines import pipeline as hf_pipeline  # fallback
    except Exception as _e:
        hf_pipeline = None
        _TRANSFORMERS_IMPORT_ERROR = _e
from typing import Optional  # Added for type hints

# Optional Gemini imports are resolved at runtime if API key is present
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # type: ignore
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False

load_dotenv()

# -------------------- Config helpers --------------------
def get_gemini_api_key() -> Optional[str]:  # Fixed type annotation
    # Prefer Streamlit secrets if present, else .env/OS env
    key = None
    try:
        # st.secrets is a Mapping, use .get for compatibility
        key = st.secrets.get("GOOGLE_API_KEY") if hasattr(st, "secrets") else None
    except Exception:
        pass
    if not key:
        key = os.getenv("GOOGLE_API_KEY")
    return key


def get_mode() -> str:
    """Return 'gemini' if an API key is provided; otherwise 'local'."""
    return "gemini" if get_gemini_api_key() else "local"


# -------------------- PDF processing --------------------
def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files locally using PyPDF2."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Error reading PDF: {getattr(pdf, 'name', 'unknown')}. Skipping. ({e})")
    return text


def get_text_chunks(text):
    """Split text into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)


# -------------------- Embeddings / Vector store --------------------
@st.cache_resource(show_spinner=False)
def get_local_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_gemini_embeddings():
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found; cannot use Gemini embeddings.")
    if not _GENAI_AVAILABLE:
        raise RuntimeError("langchain_google_genai not installed. Run: pip install google-generativeai langchain_google_genai")
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)


def get_index_dir(mode: str) -> str:
    return "faiss_index_gemini" if mode == "gemini" else "faiss_index_local"


def build_vector_store(text_chunks, mode: str):
    if mode == "gemini":
        embeddings = get_gemini_embeddings()
    else:
        embeddings = get_local_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(get_index_dir(mode))


def load_vector_store(mode: str):
    if mode == "gemini":
        embeddings = get_gemini_embeddings()
    else:
        embeddings = get_local_embeddings()
    return FAISS.load_local(
        get_index_dir(mode), embeddings, allow_dangerous_deserialization=True
    )


# -------------------- QA models --------------------
@st.cache_resource(show_spinner=False)
def get_local_qa_pipeline():
    """Return a local extractive QA pipeline, or raise a helpful error if transformers is unavailable."""
    if hf_pipeline is None:
        raise ImportError(
            "transformers.pipeline is not available. Please install/upgrade transformers:\n"
            "  pip install --upgrade transformers\n"
            "If offline, install a compatible wheel and ensure models are cached."
        )
    # Small and effective extractive QA model
    return hf_pipeline("question-answering", model="deepset/roberta-base-squad2")


@st.cache_resource(show_spinner=False)
def get_gemini_qa_chain():
    if not _GENAI_AVAILABLE:
        raise RuntimeError("langchain_google_genai not installed. Run: pip install google-generativeai langchain_google_genai")
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found; cannot use Gemini.")

    # Prompt for QA over retrieved docs
    prompt_template = (
        "Answer the question as detailed as possible from the provided context. "
        "If the answer is not in the context, say 'answer is not available in the context'.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # or gemini-1.5-pro
        temperature=0.2,
        google_api_key=api_key,
    )
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


def answer_question(user_question: str, mode: str):
    try:
        db = load_vector_store(mode)
    except Exception:
        st.error("Vector index not found. Please upload PDFs and click 'Submit & Process' first.")
        return

    docs = db.similarity_search(user_question, k=4)

    if mode == "gemini":
        chain = get_gemini_qa_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response.get("output_text", ""))
    else:
        context = "\n\n".join([d.page_content for d in docs])
        qa = get_local_qa_pipeline()
        result = qa(question=user_question, context=context)
        st.write("Reply:", result.get("answer", ""))
        score = result.get("score", 0.0)
        st.caption(f"(confidence: {score:.2f})")


# -------------------- UI --------------------
def main():
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    active_mode = get_mode()
    st.info(f"Mode: {'Gemini (cloud)' if active_mode == 'gemini' else 'Local (offline)'}")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        answer_question(user_question, active_mode)

    with st.sidebar:
        st.write("---")
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files & \n Click on the Submit & Process Button ",
            accept_multiple_files=True,
            type=["pdf"],
        )
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file first.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    build_vector_store(text_chunks, active_mode)
                    st.success("Done")
        st.write("---")
        st.write("AI App created by Shuvam Dutta")

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/Shuv13" target="_blank">Shuvam Dutta</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
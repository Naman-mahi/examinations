import streamlit as st
import PyPDF2
from transformers import pipeline
import os

st.set_page_config(page_title="RRB NTPC Exam Prep AI", layout="wide")

# --- Model choices ---
QA_MODEL = "deepset/roberta-base-squad2"  # Fast, accurate, and small
GEN_MODEL = "gpt2"  # Free, runs easily on Streamlit Cloud

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    qa = pipeline("question-answering", model=QA_MODEL)
    gen = pipeline("text-generation", model=GEN_MODEL)
    return qa, gen

qa_pipeline, text_generator = load_models()

st.title("🚆 RRB NTPC Exam Preparation AI Agent")

tabs = st.sidebar.radio("Menu", [
    "🏠 Home", 
    "📄 Upload Question Paper", 
    "📝 Generate Practice Questions", 
    "💬 Chat with AI"
])

if tabs == "🏠 Home":
    st.header("Welcome to RRB NTPC Exam Prep AI")
    st.markdown("""
    This app helps you prepare for the RRB NTPC exams (CBT 1 & 2).

    **Features:**
    - Upload previous year question papers (PDF) and ask questions about them
    - Generate practice questions (with answers) for major subjects
    - Chat with AI for explanations or extra help

    **Subjects:**
    - **Mathematics:** Number System, Percentages, Profit/Loss, Time & Work, etc.
    - **Reasoning:** Analogies, Coding-Decoding, Puzzles, etc.
    - **General Awareness:** Indian Railways, Current Affairs, History, Geography, etc.

    **Tips:**
    - Upload clear, text-based PDFs (not scanned images)
    - For best results, ask direct, specific questions
    """)

elif tabs == "📄 Upload Question Paper":
    st.header("Upload Previous Year Question Paper")
    uploaded_file = st.file_uploader("Upload a PDF question paper", type=["pdf"])
    if uploaded_file:
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_reader = PyPDF2.PdfReader("temp.pdf")
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            if text.strip():
                st.subheader("Extracted Text from PDF")
                st.text_area("PDF Content", text, height=200)
                question = st.text_input("Ask a question from the paper (e.g., 'What is the LCM of 12, 15, and 20?')")
                if question:
                    with st.spinner("AI is answering..."):
                        result = qa_pipeline(question=question, context=text)
                        answer = result['answer']
                        score = result['score']
                    st.markdown(f"**Answer:** {answer}  \n<small>(Confidence: {score:.2f})</small>", unsafe_allow_html=True)
                    if st.button("Explain this answer"):
                        prompt = f"Explain in detail: Question: {question} Answer: {answer}"
                        with st.spinner("AI is explaining..."):
                            explanation = text_generator(
                                prompt,
                                max_new_tokens=120,
                                num_return_sequences=1,
                                truncation=True,
                                pad_token_id=text_generator.tokenizer.eos_token_id if hasattr(text_generator, 'tokenizer') else None
                            )[0]['generated_text']
                        st.markdown(f"**Explanation:** {explanation}")
            else:
                st.warning("No text extracted from PDF. Make sure the PDF is text-based, not scanned images.")
            os.remove("temp.pdf")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

elif tabs == "📝 Generate Practice Questions":
    st.header("Generate Practice Questions")
    subject = st.selectbox("Select Subject", ["Mathematics", "Reasoning", "General Awareness"])
    topic = st.text_input("Enter Topic (e.g., Percentages, Coding-Decoding, Indian Railways)")
    num_questions = st.slider("Number of Questions", 1, 10, 5)
    if st.button("Generate Questions"):
        if topic:
            prompt = (
                f"Generate {num_questions} RRB NTPC {subject} practice questions on {topic}. "
                "Each question should be followed by its answer."
            )
            with st.spinner("AI is generating questions..."):
                questions = text_generator(
                    prompt,
                    max_new_tokens=256,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=text_generator.tokenizer.eos_token_id if hasattr(text_generator, 'tokenizer') else None
                )[0]['generated_text']
            st.subheader("Generated Questions")
            st.markdown(questions)
            st.download_button(
                label="Download Questions",
                data=questions,
                file_name=f"rrb_ntpc_{subject}_questions.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please enter a topic to generate questions.")

elif tabs == "💬 Chat with AI":
    st.header("Chat with AI for RRB NTPC Prep")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**AI:** {chat['ai']}")
    user_input = st.text_input("Ask anything about RRB NTPC exams, syllabus, tips…")
    if user_input:
        with st.spinner("AI is typing..."):
            prompt = (
                f"You are an expert on RRB NTPC exams. Answer helpfully and clearly.\n"
                f"User: {user_input}\nAI:"
            )
            ai_response = text_generator(
                prompt,
                max_new_tokens=120,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=text_generator.tokenizer.eos_token_id if hasattr(text_generator, 'tokenizer') else None
            )[0]['generated_text']
        st.session_state.chat_history.append({"user": user_input, "ai": ai_response})
        st.markdown(f"**You:** {user_input}")
        st.markdown(f"**AI:** {ai_response}")

st.sidebar.markdown("Built with Streamlit and free Hugging Face models (no key needed).")

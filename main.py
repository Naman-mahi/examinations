import streamlit as st
import PyPDF2
from transformers import pipeline
import os

# --- Model choices ---
QA_MODELS = {
    "DistilBERT (distilbert-base-uncased-distilled-squad)": "distilbert-base-uncased-distilled-squad",
    "RoBERTa (deepset/roberta-base-squad2)": "deepset/roberta-base-squad2",
    "BERT Large (bert-large-uncased-whole-word-masking-finetuned-squad)": "bert-large-uncased-whole-word-masking-finetuned-squad",
}
GEN_MODELS = {
    "GPT-2 (gpt2)": "gpt2",
    "Falcon 7B Instruct (tiiuae/falcon-7b-instruct)": "tiiuae/falcon-7b-instruct",
    "FLAN-T5 Base (google/flan-t5-base)": "google/flan-t5-base",
}

# --- Settings State ---
if "qa_model" not in st.session_state:
    st.session_state.qa_model = list(QA_MODELS.values())[0]
if "gen_model" not in st.session_state:
    st.session_state.gen_model = list(GEN_MODELS.values())[0]

# --- Sidebar with Settings tab ---
tabs = st.sidebar.radio("Menu", ["Home", "Upload Question Paper", "Generate Practice Questions", "Chat with AI", "Settings"])
if tabs == "Settings":
    st.header("AI Model Settings")
    st.markdown("Choose which free AI models to use for answering questions and generating explanations or practice questions.")
    qa_model_display = list(QA_MODELS.keys())
    gen_model_display = list(GEN_MODELS.keys())

    qa_selected = st.selectbox(
        "Question Answering Model",
        qa_model_display,
        index=qa_model_display.index(
            [k for k, v in QA_MODELS.items() if v == st.session_state.qa_model][0]
        ),
    )
    gen_selected = st.selectbox(
        "Text Generation Model",
        gen_model_display,
        index=gen_model_display.index(
            [k for k, v in GEN_MODELS.items() if v == st.session_state.gen_model][0]
        ),
    )
    st.session_state.qa_model = QA_MODELS[qa_selected]
    st.session_state.gen_model = GEN_MODELS[gen_selected]

    st.write("**Free Question Answering Models:**")
    for k, v in QA_MODELS.items():
        st.write(f"- `{v}` ({k})")
    st.write("**Free Text Generation Models:**")
    for k, v in GEN_MODELS.items():
        st.write(f"- `{v}` ({k})")
    st.stop()

# --- Load models based on settings ---
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipelines(qa_model_name, gen_model_name):
    qa = pipeline("question-answering", model=qa_model_name)
    gen = pipeline("text-generation", model=gen_model_name)
    return qa, gen

qa_pipeline, generator = load_pipelines(st.session_state.qa_model, st.session_state.gen_model)

# Streamlit app configuration
st.set_page_config(page_title="RRB NTPC Exam Prep AI", layout="wide")
st.title("RRB NTPC Exam Preparation AI Agent")

# Sidebar for navigation (redundant because tabs are used, but kept for your layout)
# st.sidebar.title("Navigation")

# Home page: Display syllabus and instructions
if tabs == "Home":
    st.header("Welcome to RRB NTPC Exam Prep AI")
    st.markdown("""
    This free AI-powered app helps you prepare for the RRB NTPC exams (CBT 1 & CBT 2).
    **Features**:
    - Upload previous year question paper PDFs to get answers.
    - Generate practice questions for Mathematics, Reasoning, and General Awareness.
    - Chat with the AI for explanations or additional questions.
    **Syllabus**:
    - **Mathematics**: Number System, Percentages, Profit/Loss, Time & Work, etc.
    - **General Intelligence & Reasoning**: Analogies, Coding-Decoding, Puzzles, etc.
    - **General Awareness**: Indian Railways, Current Affairs, History, Geography, etc.
    **Instructions**:
    - Use the sidebar to navigate.
    - Upload clear, text-based PDFs for accurate question extraction.
    - Input questions clearly in the Chat section for best results.
    """)

# Upload Question Paper: Extract and answer questions from PDFs
elif tabs == "Upload Question Paper":
    st.header("Upload Previous Year Question Paper")
    uploaded_file = st.file_uploader("Upload a PDF question paper", type=["pdf"])
    
    if uploaded_file:
        # Save uploaded PDF temporarily
        try:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader("temp.pdf")
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            if text.strip():
                st.subheader("Extracted Text from PDF")
                st.text_area("PDF Content", text, height=200)
                
                # Input question to answer based on extracted text
                question = st.text_input("Enter a question from the paper (e.g., 'What is the LCM of 12, 15, and 20?')")
                if question:
                    # Use DistilBERT to answer the question
                    result = qa_pipeline(question=question, context=text)
                    st.markdown(f"**Answer**: {result['answer']} (Confidence: {result['score']:.2f})")
                    
                    # Offer explanation
                    if st.button("Explain this answer"):
                        explanation_prompt = f"Explain how to solve this question: {question} Answer: {result['answer']}"
                        with st.spinner("Generating explanation..."):
                            explanation = generator(
                                explanation_prompt,
                                max_length=150,
                                num_return_sequences=1,
                                truncation=True,
                                pad_token_id=generator.tokenizer.eos_token_id if hasattr(generator, 'tokenizer') else None
                            )[0]['generated_text']
                        st.markdown(f"**Explanation**: {explanation}")
            else:
                st.warning("No text extracted from PDF. Ensure the PDF is text-based, not scanned.")
            
            # Clean up temporary file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
                
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# Generate Practice Questions: Create syllabus-based questions
elif tabs == "Generate Practice Questions":
    st.header("Generate Practice Questions")
    subject = st.selectbox("Select Subject", ["Mathematics", "General Intelligence & Reasoning", "General Awareness"])
    topic = st.text_input("Enter Topic (e.g., Percentages, Coding-Decoding, Indian Railways)")
    num_questions = st.slider("Number of Questions", 1, 10, 5)

    if st.button("Generate Questions"):
        if topic:
            # Create prompt for question generation
            prompt = f"Generate {num_questions} RRB NTPC {subject} practice questions on {topic} with answers."
            with st.spinner("Generating questions..."):
                questions = generator(
                    prompt,
                    max_length=500,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=generator.tokenizer.eos_token_id if hasattr(generator, 'tokenizer') else None
                )[0]['generated_text']
            
            # Display questions
            st.subheader("Generated Questions")
            st.markdown(questions)
            
            # Allow download of generated questions
            st.download_button(
                label="Download Questions",
                data=questions,
                file_name=f"rrb_ntpc_{subject}_questions.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please enter a topic to generate questions.")

# Chat with AI: Interactive Q&A for explanations or additional queries
elif tabs == "Chat with AI":
    st.header("Chat with AI for RRB NTPC Prep")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"**You**: {chat['question']}")
        st.markdown(f"**AI**: {chat['answer']}")
    
    # Input new question
    user_question = st.text_input("Ask a question or request an explanation (e.g., 'Explain how to solve LCM questions' or 'What is the longest railway platform in India?')")
    if user_question:
        # Use DistilBERT for answering
        context = (
            "RRB NTPC exam preparation context: Covers Mathematics (LCM, Percentages, etc.), "
            "General Intelligence & Reasoning (Coding-Decoding, Puzzles), "
            "General Awareness (Indian Railways, Current Affairs, History, Geography). Provide accurate answers."
        )
        with st.spinner("Processing your question..."):
            result = qa_pipeline(question=user_question, context=context)
            answer = result['answer']
        
        # Append to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        
        # Display latest response
        st.markdown(f"**You**: {user_question}")
        st.markdown(f"**AI**: {answer}")
        
        # Offer detailed explanation
        if st.button("Explain this answer"):
            explanation_prompt = f"Explain in detail: {user_question} Answer: {answer}"
            with st.spinner("Generating explanation..."):
                explanation = generator(
                    explanation_prompt,
                    max_length=150,
                    num_return_sequences=1,
                    truncation=True,
                    pad_token_id=generator.tokenizer.eos_token_id if hasattr(generator, 'tokenizer') else None
                )[0]['generated_text']
            st.markdown(f"**Explanation**: {explanation}")

# Footer
st.sidebar.markdown("Built with Streamlit and Hugging Face. Deploy locally or on Streamlit Community Cloud.")
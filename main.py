import streamlit as st
import PyPDF2
from transformers import pipeline
import os

# Initialize Hugging Face pipelines
# Question-answering model (DistilBERT for solving questions)
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
# Text generation model (GPT-2 for generating practice questions)
generator = pipeline("text-generation", model="gpt2")

# Streamlit app configuration
st.set_page_config(page_title="RRB NTPC Exam Prep AI", layout="wide")
st.title("RRB NTPC Exam Preparation AI Agent")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a feature", ["Home", "Upload Question Paper", "Generate Practice Questions", "Chat with AI"])

# Home page: Display syllabus and instructions
if page == "Home":
    st.header("Welcome to RRB NTPC Exam Prep AI")
    st.write("""
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
    - Upload clear PDFs for accurate question extraction.
    - For best results, input questions clearly in the Chat section.
    """)

# Upload Question Paper: Extract and answer questions from PDFs
elif page == "Upload Question Paper":
    st.header("Upload Previous Year Question Paper")
    uploaded_file = st.file_uploader("Upload a PDF question paper", type=["pdf"])
    
    if uploaded_file:
        # Save uploaded PDF temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Extract text from PDF
        try:
            pdf_reader = PyPDF2.PdfReader("temp.pdf")
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            
            st.subheader("Extracted Text from PDF")
            st.text_area("PDF Content", text, height=200)
            
            # Input question to answer based on extracted text
            question = st.text_input("Enter a question from the paper (e.g., 'What is the LCM of 12, 15, and 20?')")
            if question:
                # Use DistilBERT to answer the question based on PDF text
                result = qa_pipeline(question=question, context=text)
                st.write(f**Answer**: {result['answer']} (Confidence: {result['score']:.2f})")
                # Offer explanation
                if st.button("Explain this answer"):
                    explanation_prompt = f"Explain how to solve this question: {question} Answer: {result['answer']}"
                    explanation = generator(explanation_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
                    st.write(f**Explanation**: {explanation}")
        
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
        
        # Clean up temporary file
        if os.path.exists("temp.pdf"):
            os.remove("temp.pdf")

# Generate Practice Questions: Create syllabus-based questions
elif page == "Generate Practice Questions":
    st.header("Generate Practice Questions")
    subject = st.selectbox("Select Subject", ["Mathematics", "General Intelligence & Reasoning", "General Awareness"])
    topic = st.text_input("Enter Topic (e.g., Percentages, Coding-Decoding, Indian Railways)")
    num_questions = st.slider("Number of Questions", 1, 10, 5)
    
    if st.button("Generate Questions"):
        # Create prompt for question generation
        prompt = f"Generate {num_questions} RRB NTPC {subject} practice questions on {topic} with answers."
        questions = generator(prompt, max_length=500, num_return_sequences=1)[0]['generated_text']
        
        # Display questions
        st.subheader("Generated Questions")
        st.write(questions)
        
        # Allow download of generated questions
        st.download_button(
            label="Download Questions",
            data=questions,
            file_name=f"rrb_ntpc_{subject}_questions.txt",
            mime="text/plain"
        )

# Chat with AI: Interactive Q&A for explanations or additional queries
elif page == "Chat with AI":
    st.header("Chat with AI for RRB NTPC Prep")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for chat in st.session_state.chat_history:
        st.write(f**You**: {chat['question']}")
        st.write(f**AI**: {chat['answer']}")
    
    # Input new question
    user_question = st.text_input("Ask a question or request an explanation (e.g., 'Explain how to solve LCM questions' or 'What is the longest railway platform in India?')")
    if user_question:
        # Use DistilBERT for general knowledge or explanation
        context = "RRB NTPC exam preparation context: Covers Mathematics (LCM, Percentages, etc.), General Intelligence & Reasoning (Coding-Decoding, Puzzles), General Awareness (Indian Railways, Current Affairs, History, Geography). Provide accurate answers."
        result = qa_pipeline(question=user_question, context=context)
        answer = result['answer']
        
        # Append to chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})
        
        # Display latest response
        st.write(f**You**: {user_question}")
        st.write(f**AI**: {answer}")
        
        # Offer detailed explanation
        if st.button("Explain this answer"):
            explanation_prompt = f"Explain in detail: {user_question} Answer: {answer}"
            explanation = generator(explanation_prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
            st.write(f**Explanation**: {explanation}")

# Footer
st.sidebar.write("Built with Streamlit and Hugging Face. Deploy locally or on Streamlit Community Cloud.")

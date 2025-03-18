import os
import streamlit as st
import time
import smtplib
import mimetypes
from email.message import EmailMessage
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("profile.txt")  # Ensure this file exists in your Space
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)  
docs = text_splitter.split_documents(documents)

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Configure retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Load LLM model with token limits
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    model_type="mistral",
    max_new_tokens=256,
    temperature=0.7
)

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Set Streamlit page config
st.set_page_config(page_title="AskYashika", page_icon="🤖", layout="centered")

st.markdown('<h1 style="text-align:center;">🚀 AskYashika</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:gray;">Not Into Scanning Resumes? Let\'s Chat About the Good Stuff!</p>', unsafe_allow_html=True)

# Input Field for query
query = st.text_input("💡 Type your question here:", placeholder="E.g., What projects has Yashika worked on?")

# Define custom responses based on predefined topics
def custom_responses(query):
    career_keywords = ["career", "journey", "internships", "experience"]
    project_keywords = ["projects", "AI projects", "machine learning", "web development"]
    skills_keywords = ["skills", "technologies", "tools", "programming languages"]
    yashika_keywords = ["who is yashika", "what is yashika", "tell me about yashika", "about yashika"]

    if any(keyword in query.lower() for keyword in yashika_keywords):
        return "Yashika Jain is a Computer Science Engineering student with a strong passion for AI and Data Science. She has interned with companies like Techwalnut, Prodigy Infotech, and Zscaler."

    elif any(keyword in query.lower() for keyword in career_keywords):
        return "Yashika has worked on multiple impactful projects, including AI-driven stock price prediction and early stroke detection systems."

    elif any(keyword in query.lower() for keyword in project_keywords):
        return "Yashika has developed AI-powered chatbots, stock price predictors, and real-time weather apps using Flask."

    elif any(keyword in query.lower() for keyword in skills_keywords):
        return "Yashika is proficient in Python, Java, Golang, TensorFlow, Scikit-learn, AWS, and React.js."

    else:
        return None

# Process query and return response
if query:
    with st.spinner("🤖 Thinking... Hold on tight!"):
        time.sleep(2)  # Simulating processing time
        
        custom_answer = custom_responses(query)
        
        if custom_answer is None:
            response = qa.invoke(query)
            answer = response.get('result', '') if isinstance(response, dict) else response
            st.markdown(f'<div style="background-color:#333;padding:10px;border-radius:5px;color:white;">{answer}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background-color:#333;padding:10px;border-radius:5px;color:white;">{custom_answer}</div>', unsafe_allow_html=True)

# 📩 Resume Email Feature
def send_resume(email):
    sender_email = os.getenv("EMAIL_USER")  # Fetch email from Railway/Secrets
    sender_password = os.getenv("EMAIL_PASS")  # Fetch app password securely
    receiver_email = email
    subject = "Yashika's Resume for Your Reference"
    body = "Hello,\n\nPlease find attached Yashika's resume for your reference.\n\nBest regards,\nAskYashika AI Agent"

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email

    resume_path = "Yashika_Resume.pdf"  # Ensure this file is uploaded in Railway
    if os.path.exists(resume_path):
        mime_type, _ = mimetypes.guess_type(resume_path)
        mime_type = mime_type or "application/pdf"
        with open(resume_path, "rb") as file:
            msg.add_attachment(file.read(), maintype="application", subtype="pdf", filename="Yashika_Resume.pdf")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "✅ Resume sent successfully!"
    except Exception as e:
        return f"❌ Failed to send email: {str(e)}"

# Resume Email UI
st.subheader("📩 Get Yashika's Resume via Email")
user_email = st.text_input("Enter your email to receive Yashika's resume:")
if st.button("Send Resume"):
    if user_email:
        response = send_resume(user_email)
        st.success(response)
    else:
        st.warning("⚠️ Please enter a valid email address.")

# 🎶 Vibe and Chat Section
st.markdown("### 🎶 Want to vibe while chatting?")
st.markdown("[Play My Vibe Song](https://www.youtube.com/watch?v=3JZ_D3ELwOQ)", unsafe_allow_html=True)

import sys
if __name__ == "__main__":
    sys.argv.append(f"--server.port={port}")
    sys.argv.append("--server.address=0.0.0.0")  # Allow external access
    st._run()




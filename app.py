import os
import time
import smtplib
import mimetypes
import gradio as gr
from email.message import EmailMessage
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import CTransformers

# Load and split documents
loader = TextLoader("profile.txt")  # Ensure this file exists
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Configure retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Load LLM model
llm = CTransformers(
    model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",  
    model_type="mistral",
    max_new_tokens=256,
    temperature=0.7
)

# Create retrieval-based QA chain
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

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

# AI Agent Function
def ask_yashika(query):
    time.sleep(2)  # Simulating processing time
    custom_answer = custom_responses(query)
    
    if custom_answer is None:
        response = qa.invoke(query)
        answer = response.get('result', '') if isinstance(response, dict) else response
    else:
        answer = custom_answer

    return answer

# Function to send resume
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

    resume_path = "Yashika_Resume.pdf"  # Ensure this file is uploaded
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

# Gradio UI
with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align:center;'>🚀 AskYashika</h1>")
    gr.HTML("<p style='text-align:center; color:gray;'>Not Into Scanning Resumes? Let's Chat About the Good Stuff!</p>")

    with gr.Row():
        query_input = gr.Textbox(label="💡 Type your question here:", placeholder="E.g., What projects has Yashika worked on?")
        response_output = gr.Textbox(label="🤖 AskYashika's Response", interactive=False)

    query_button = gr.Button("Ask 🚀")
    query_button.click(fn=ask_yashika, inputs=query_input, outputs=response_output)

    gr.HTML("<h3>📩 Get Yashika's Resume via Email</h3>")
    email_input = gr.Textbox(label="Enter your email:")
    email_button = gr.Button("Send Resume")
    email_output = gr.Textbox(label="📩 Email Status", interactive=False)
    
    email_button.click(fn=send_resume, inputs=email_input, outputs=email_output)

    gr.HTML("<h3>🎶 Want to vibe while chatting?</h3>")
    gr.HTML('<a href="https://www.youtube.com/watch?v=VuNIsY6JdUw" target="_blank">🎵 Play My Vibe Song</a>')

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()

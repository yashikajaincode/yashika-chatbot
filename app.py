import streamlit as st
import time
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("profile.txt")  # Assume profile.txt contains your career story and details
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)  # Optimized chunking
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

# Streamlit UI Styling
st.set_page_config(page_title="AskYashika", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
        body {
            background-color: #1E1E1E;
            color: white;
            font-family: 'Poppins', sans-serif;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 1rem;
            text-align: center;
            margin-bottom: 30px;
            color: #bbbbbb;
        }
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            padding: 10px;
        }
        .loading {
            font-size: 1rem;
            color: #ffcc00;
            text-align: center;
            font-weight: bold;
        }
        .chat-response {
            background-color: #333;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-size: 1.1rem;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🚀 AskYashika </div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Not Into Scanning Resumes? Let\'s Chat About the Good Stuff! (⚠️ Still in development, responses may take time)</div>', unsafe_allow_html=True)

# Input Field for query
query = st.text_input("💡 Type your question here:", placeholder="E.g., What projects has Yashika worked on?")

# Define custom responses based on predefined topics
def custom_responses(query):
    # Keywords for Career-related topics
    career_keywords = ["career", "journey", "internships", "experience"]
    project_keywords = ["projects", "AI projects", "machine learning", "web development"]
    skills_keywords = ["skills", "technologies", "tools", "programming languages"]
    yashika_keywords = ["who is yashika", "what is yashika", "tell me about yashika", "about yashika"]
    
    # Check if the query is about Yashika's profile
    if any(keyword in query.lower() for keyword in yashika_keywords):
        return "Yashika Jain is a Computer Science Engineering student at Shri Ramdeobaba College of Engineering & Management. She has a strong interest in AI technologies and data science. Yashika has worked on AI-based projects, including stock price prediction and early stroke detection. She has interned with companies like Techwalnut, Prodigy Infotech, and Zscaler. She is also a member of the Rotaract Club of RCOEM and the Data Science Association."

    # Check if the query relates to career
    elif any(keyword in query.lower() for keyword in career_keywords):
        return "Yashika has worked on multiple impactful projects, including AI-driven stock price prediction and early stroke detection systems. Her journey includes internships at Techwalnut, Prodigy Infotech, and Zscaler."

    # Check if the query relates to projects
    elif any(keyword in query.lower() for keyword in project_keywords):
        return "Yashika's notable projects include building AI-powered chatbots with LangChain, creating a weather app using Flask, and integrating ML algorithms to predict stock prices."

    # Check if the query relates to skills
    elif any(keyword in query.lower() for keyword in skills_keywords):
        return "Yashika is proficient in programming languages like Python, Java, and Golang. She also works with tools like TensorFlow, Scikit-learn, AWS, and frameworks like React and Flask."
    
    # If the query does not match predefined topics, use QA system
    else:
        return None

# Song Recommendation or Play
song_url = "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"  # Example YouTube link (you can replace with any song URL)

# Main logic to process query and return response
if query:
    with st.spinner("🤖 Thinking... Hold on tight!"):
        time.sleep(2)  # Simulating processing time
        
        # Try to get a custom response based on predefined topics
        custom_answer = custom_responses(query)
        
        # If no custom answer, use QA system
        if custom_answer is None:
            response = qa.invoke(query)
            # Extract the actual answer from the response (to avoid displaying query + result)
            answer = response.get('result', '') if isinstance(response, dict) else response
            # Display only the response (no query displayed)
            st.markdown(f'<div class="chat-response">{answer}</div>', unsafe_allow_html=True)
        else:
            # Display custom response (no query displayed)
            st.markdown(f'<div class="chat-response">{custom_answer}</div>', unsafe_allow_html=True)

# Vibe and Chat Section
st.markdown("### 🎶 Want to vibe while chatting?")
st.markdown("Let the music set the mood for our chat!")
st.markdown("[Play My Vibe Song](%s)" % song_url, unsafe_allow_html=True)  # Link to play a song

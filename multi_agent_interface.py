import streamlit as st
import os
import json
import re
from typing import Dict, List, Any
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Set page configuration
st.set_page_config(
    page_title="ESI Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input, .stTextArea>div>textarea {
        background-color: #ffffff;
    }
    .output-container {
        background-color: black;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    h1 {
        color: #1e88e5;
    }
    h2, h3 {
        color: #0d47a1;
    }
    .stButton>button {
        background-color: #1e88e5;
        color: white;
    }
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .agent-message {
        background-color: black;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Import environment variables
load_dotenv()  # Load environment variables from .env file

# API Keys loaded from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Check if keys are available
if not GROQ_API_KEY or not GEMINI_API_KEY:
    st.error("API keys not found. Please check your .env file.")

# Set up API keys
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Function to load the FAISS index
@st.cache_resource
def load_faiss_index():

    
    try:
        # Initialize embeddings
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True
            }
        )
        
        # Load the vector store from the ./faiss_index directory
        faiss_index_path = "./faiss_index"
        if not os.path.exists(faiss_index_path):
            faiss_index_path = "./dd/faiss_index"  # Alternative path

        if not os.path.exists(faiss_index_path):
            st.error(f"FAISS index not found at {faiss_index_path}. Please check the index location.")
            return None
            
        db = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return None

# Function to create GROQ retriever agent
def create_groq_agent(db):
    if db is None:
        return None
        
    try:
        # Initialize the GROQ LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",  # Using the model from ggg.py
            temperature=0,
            max_tokens=32768
        )
        
        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_template('''
        [Context] {context}
        [Question] {input}
        
        As an ESI educational assistant, respond to this query with accurate information.
        Base your answer on the provided context and your knowledge.
        
        When discussing programming or algorithms, include code examples where appropriate.
        Structure your response with clear headings and use bullet points for key concepts.
        ''')
        
        # Create the retriever
        retriever = db.as_retriever(search_kwargs={"k": 6})  # Retrieve top 6 most relevant docs
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
    except Exception as e:
        st.error(f"Error creating GROQ agent: {str(e)}")
        return None

# Function to create Gemini enhancer agent
def create_gemini_agent():
    try:
        # Initialize Gemini model
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")  # Using the model from ggg.py
        return model
    except Exception as e:
        st.error(f"Error creating Gemini agent: {str(e)}")
        return None

# Function to process queries with the multi-agent system
def process_query(query: str, groq_chain, gemini_model):
    if groq_chain is None or gemini_model is None:
        return "Error: One or more AI models failed to initialize. Please check the error messages."
        
    try:
        # Step 1: Get initial response from GROQ
        groq_result = groq_chain.invoke({'input': query})
        groq_response = groq_result['answer'] if 'answer' in groq_result else "No response generated."
        
        # Step 2: Enhance with Gemini
        gemini_prompt = f"""
        You are an expert ESI (École Supérieure d'Informatique) instructor specializing in computer science education.
        
        Base information provided by another AI system:
        {groq_response}
        
        User query: {query}
        
        Your task is to:
        1. Validate the accuracy of the information provided
        2. Correct any factual errors or misconceptions
        3. Enhance the response with additional context, examples, or clarifications
        4. Improve the structure and clarity of the explanation
        5. Add code examples if relevant (especially for programming topics)
        
        Provide your enhanced response using appropriate academic language and structure.
        If you make significant corrections, briefly note what was corrected.
        Use markdown formatting for clear presentation.
        """
        
        gemini_response = gemini_model.generate_content(gemini_prompt)
        enhanced_response = gemini_response.text
        
        return {
            "groq_response": groq_response,
            "enhanced_response": enhanced_response
        }
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Main Streamlit app
def main():
    # Sidebar - Settings and controls
    with st.sidebar:
        st.title("ESI Multi-Agent System")
        st.markdown("### Settings")
        
        # Language selection
        language = st.radio("Language:", ("English", "Français"))
        
        # System information
        st.markdown("---")
        st.markdown("### System Information")
        st.markdown("""
        This system uses a multi-agent approach:
        1. **GROQ Agent**: Retrieves relevant information using vector search
        2. **Gemini Agent**: Validates and enhances the information
        
        The system specializes in ESI academic subjects, including:
        - Programming (C, Assembly, Algorithms)
        - Data Structures
        - Z Language
        - Computer Architecture
        """)
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")

    # Main content area
    st.title("ESI Academic Assistant")
    
    # Initialize the agents
    db = load_faiss_index()
    groq_chain = create_groq_agent(db)
    gemini_model = create_gemini_agent()
    
    if None in [db, groq_chain, gemini_model]:
        st.warning("Some components failed to load. The assistant may not function correctly.")
    
    # Chat interface
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <b>You:</b><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message agent-message">
                <b>Assistant:</b><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_area("Your question:", key="user_query", 
                               placeholder="Ask a question about ESI subjects, programming, algorithms, etc.")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_query:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user-message">
            <b>You:</b><br>{user_query}
        </div>
        """, unsafe_allow_html=True)
        
        # Process the query
        with st.spinner("Thinking..."):
            result = process_query(user_query, groq_chain, gemini_model)
            
            if isinstance(result, dict):
                # Add system response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": result["enhanced_response"]
                })
                
                # Display system response
                st.markdown(f"""
                <div class="chat-message agent-message">
                    <b>Assistant:</b><br>{result["enhanced_response"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Option to show the raw GROQ response
                with st.expander("Show raw response"):
                    st.markdown(result["groq_response"])
            else:
                # Add error message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": f"Error: {result}"
                })
                
                # Display error message
                st.error(result)

if __name__ == "__main__":
    main()

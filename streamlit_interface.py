import streamlit as st
import os
import time
from tryllm import query_system, genai

# Page configuration
st.set_page_config(
    page_title="ESI Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main {
        background-color: #0e1117; 
    }
    .chat-message {
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-left: 5px solid #4b91f1;
    }
    .chat-message.assistant {
        background-color: #1e2730;
        border-left: 5px solid #26a69a;
    }
    .chat-message .avatar {
        width: 20%;
        display: flex;
        justify-content: center;
        align-items: flex-start;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .content {
        width: 80%;
        padding-left: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e1e1e;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: #80cbc4;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e1e1e !important;
        color: white !important;
        border-bottom: 2px solid #26a69a !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False
# New flag to control input clearing
if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

def display_chat_message(role, content, avatar_url=None):
    """Display a chat message with styling based on the role."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">
                <img src="https://www.iconpacks.net/icons/2/free-user-icon-3296-thumb.png" alt="User Avatar">
            </div>
            <div class="content">
                <p><strong>You</strong></p>
                <p>{content}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant">
            <div class="avatar">
                <img src="https://www.iconpacks.net/icons/2/free-robot-icon-1768-thumb.png" alt="Assistant Avatar">
            </div>
            <div class="content">
                <p><strong>ESI Assistant</strong></p>
                <p>{content}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def process_query(query):
    """Process the user query through both GROQ and Gemini models."""
    st.session_state.processing = True
    
    # Step 1: Get initial response from GROQ
    with st.spinner("Retrieving information..."):
        try:
            groq_response = query_system(query)
        except Exception as e:
            st.error(f"Error with GROQ processing: {str(e)}")
            return None, str(e)
    
    # Step 2: Enhance with Gemini
    with st.spinner("Enhancing response..."):
        try:
            model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
            gemini_prompt = f"""
            -a user asked this question: {query}\n
            -and an AI assistant gave me this response
            {groq_response}\n
            -please do the following: 
            - Enhance the response with academic rigor and clarity
            - If the response has mathematical formulas, ensure they are formatted correctly
            - Verify the logic of any algorithms, programs, or methods described
            - Structure the response in clear sections with headings where appropriate
            - Format the output as markdown for better readability
            - Provide examples where helpful
            """
            
            gemini_response = model.generate_content(gemini_prompt)
            enhanced_response = gemini_response.text
        except Exception as e:
            st.error(f"Error with Gemini processing: {str(e)}")
            enhanced_response = groq_response
    
    st.session_state.processing = False
    return groq_response, enhanced_response

# Sidebar
with st.sidebar:
    st.title("ESI Multi-Agent System")
    st.markdown("---")
    st.markdown("""
    ### How it works
    
    This interface connects to a multi-agent system that:
    
    1. **GROQ Agent**: Retrieves relevant information using vector search with FAISS
    2. **Gemini Agent**: Validates and enhances the information
    
    ### Topics
    
    Ask about:
    - Z Language
    - Assembly (8086)
    - Algorithms and data structures
    - Programming concepts
    - Computer architecture
    - Mathematics
    """)
    
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()  # Changed from st.experimental_rerun()

# Main content area
st.title("ESI Academic Multi-Agent Assistant")

# Display chat history
for message in st.session_state.messages:
    display_chat_message(message["role"], message["content"])

# User input area
with st.container():
    # Use a callback to prepare input clearing for next render cycle
    def on_submit():
        st.session_state.clear_input = True
    
    # Check if we need to provide an empty default value
    default_value = "" if st.session_state.clear_input else st.session_state.get("user_input_value", "")
    
    user_input = st.text_area(
        "Your question:", 
        key="user_input",
        height=100,
        value=default_value
    )
    
    # Reset the clear flag after rendering with empty value
    if st.session_state.clear_input:
        st.session_state.clear_input = False
    
    # Process user input
    if st.button("Send") and user_input and not st.session_state.processing:
        # Store the current input value
        st.session_state.user_input_value = user_input
        
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response from the multi-agent system
        groq_response, enhanced_response = process_query(user_input)
        
        # Add system response to chat
        if enhanced_response:
            st.session_state.messages.append({"role": "assistant", "content": enhanced_response})
        
        # Store raw response if needed
        if "raw_responses" not in st.session_state:
            st.session_state.raw_responses = []
        if groq_response:
            st.session_state.raw_responses.append(groq_response)
        
        # Set flag to clear input on next rerun
        st.session_state.clear_input = True
        
        # Rerun the app to update the chat display and clear input
        st.rerun()  # Changed from st.experimental_rerun()

# Advanced section (tabs)
if st.session_state.messages:
    st.markdown("---")
    tabs = st.tabs(["Enhanced Response", "Raw GROQ Response", "Response Analysis"])
    
    with tabs[0]:
        st.markdown("### Current Response (Enhanced by Gemini)")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.markdown(st.session_state.messages[-1]["content"])
    
    with tabs[1]:
        st.markdown("### Raw GROQ Response")
        if len(st.session_state.messages) >= 2 and st.session_state.messages[-1]["role"] == "assistant":
            # Store raw GROQ response in session state
            if "raw_responses" not in st.session_state:
                st.session_state.raw_responses = []
            
            if len(st.session_state.raw_responses) > 0:
                st.markdown(st.session_state.raw_responses[-1])
            else:
                st.info("No raw GROQ response available for this conversation.")
            
    with tabs[2]:
        st.markdown("### Response Analysis")
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            response = st.session_state.messages[-1]["content"]
            # Count characters, words, and estimate response quality
            char_count = len(response)
            word_count = len(response.split())
            has_code = "```" in response
            has_lists = "- " in response or "* " in response
            has_headings = "#" in response
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", char_count)
            with col2:
                st.metric("Words", word_count)
            with col3:
                quality = "High" if has_code and has_lists and has_headings else "Medium" if (has_code or has_lists or has_headings) else "Basic"
                st.metric("Format Quality", quality)
            
            st.markdown("**Content Features:**")
            st.write(f"- Contains code blocks: {'✅' if has_code else '❌'}")
            st.write(f"- Contains lists: {'✅' if has_lists else '❌'}")
            st.write(f"- Contains headings: {'✅' if has_headings else '❌'}")

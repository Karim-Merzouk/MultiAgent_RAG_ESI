import streamlit as st
import os
import json
import traceback
import time
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import google.generativeai as genai

# Configure API keys
load_dotenv()  # Load environment variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Import prompt templates from rag_backend
from rag_backend import (
    ESI_PROMPT_TEMPLATES,
    SUBJECT_PROMPTS,
    AGENT_SYSTEM_PROMPTS,
    ESI_TOPICS,
    detect_user_intent,
    detect_subject_area
)

class Agent:
    """Base class for all agents in the multi-agent system"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.system_prompt = AGENT_SYSTEM_PROMPTS.get(role, "You are a helpful assistant.")
        self.conversation_history = []
    
    def add_to_history(self, message: Dict[str, Any]):
        """Add a message to this agent's conversation history"""
        self.conversation_history.append(message)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get this agent's conversation history"""
        return self.conversation_history
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return a response"""
        raise NotImplementedError("Subclasses must implement process method")

class CoordinatorAgent(Agent):
    """Agent responsible for coordinating the multi-agent workflow"""
    
    def __init__(self):
        super().__init__("Coordinator", "coordinator")
        self.groq_agent = None
        self.gemini_agent = None
    
    def register_agents(self, groq_agent, gemini_agent):
        """Register sub-agents with the coordinator"""
        self.groq_agent = groq_agent
        self.gemini_agent = gemini_agent
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine the best processing strategy"""
        # Use the detection functions from rag_backend
        intent = detect_user_intent(query)
        subject = detect_subject_area(query)
        
        return {
            "query": query,
            "detected_intent": intent,
            "detected_subject": subject,
            "requires_code_analysis": any(term in query.lower() for term in ["code", "analyze", "debug", "implement", "programming"])
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the user query through the multi-agent workflow"""
        query = input_data.get("query", "")
        language = input_data.get("language", "en")
        
        # Step 1: Analyze the query
        analysis = self.analyze_query(query)
        self.add_to_history({"role": "system", "content": f"Query analysis: {json.dumps(analysis)}"})
        
        # Step 2: Have Groq agent process the query with retrieval
        groq_input = {
            "query": query,
            "intent": analysis["detected_intent"],
            "subject": analysis["detected_subject"],
            "language": language
        }
        groq_response = self.groq_agent.process(groq_input)
        self.add_to_history({"role": "groq_agent", "content": groq_response})
        
        # Step 3: Have Gemini agent validate and enhance the response
        gemini_input = {
            "query": query,
            "groq_response": groq_response,
            "intent": analysis["detected_intent"],
            "subject": analysis["detected_subject"],
            "language": language
        }
        gemini_response = self.gemini_agent.process(gemini_input)
        self.add_to_history({"role": "gemini_agent", "content": gemini_response})
        
        # Step 4: Final synthesis and response generation
        final_response = self.synthesize_response(query, groq_response, gemini_response, analysis)
        
        return {
            "response": final_response,
            "analysis": analysis,
            "groq_response": groq_response,
            "gemini_response": gemini_response
        }
    
    def synthesize_response(self, query: str, groq_response: Dict[str, Any], 
                           gemini_response: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize the final response from agent outputs"""
        # Log the synthesis process
        self.add_to_history({
            "role": "coordinator", 
            "content": "Synthesizing final response based on Groq retrieval and Gemini validation"
        })
        
        # Extract the most relevant parts from both responses
        groq_answer = groq_response.get("answer", "")
        gemini_answer = gemini_response.get("enhanced_response", "")
        
        # Simple metrics for response quality
        metrics = {
            "groq_confidence": groq_response.get("confidence", 0.8),
            "gemini_confidence": gemini_response.get("confidence", 0.9),
            "response_length": len(gemini_answer),
            "factual_corrections": gemini_response.get("corrections_count", 0),
            "enhancement_count": gemini_response.get("enhancements_count", 0)
        }
        
        # Choose the best response based on our analysis
        chosen_response = gemini_answer if gemini_answer else groq_answer
        
        return {
            "content": chosen_response,
            "metrics": metrics,
            "source": "gemini" if gemini_answer else "groq",
            "subject": analysis["detected_subject"],
            "intent": analysis["detected_intent"]
        }

class GroqRetrieverAgent(Agent):
    """Agent using Groq for retrieval and initial response generation"""
    
    def __init__(self, embeddings, vector_store):
        super().__init__("Groq Retriever", "groq_retriever")
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 7})
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=8192
        )
        
    def get_prompt_template(self, intent=None, subject=None):
        """Get appropriate prompt template based on intent and subject"""
        # First check subject-specific prompts (they take precedence)
        if subject and subject in SUBJECT_PROMPTS:
            return SUBJECT_PROMPTS[subject]
            
        # Then check intent-specific prompts
        if intent and intent in ESI_PROMPT_TEMPLATES:
            return ESI_PROMPT_TEMPLATES[intent]
            
        # Default to concept explanation if nothing found
        prompt = ESI_PROMPT_TEMPLATES.get("concept_explanation", None)
        
        # If still None (shouldn't happen), create a basic template
        if prompt is None:
            prompt = ChatPromptTemplate.from_template('''
            [Context] {context}
            [Question] {input}
            
            Please answer this question based on the provided context and your knowledge.
            ''')
        
        return prompt
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the query using RAG with Groq"""
        query = input_data.get("query", "")
        intent = input_data.get("intent", "concept_explanation")
        subject = input_data.get("subject")
        language = input_data.get("language", "en")
        
        # Add language instruction if needed
        query_for_llm = query
        if language.lower() == "fr":
            query_for_llm = f"[Répondre en français s'il vous plaît] {query}"
        
        # Get appropriate prompt template
        prompt_template = self.get_prompt_template(intent, subject)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt_template)
        retrieval_chain = create_retrieval_chain(self.retriever, document_chain)
        
        # Execute the chain
        start_time = time.time()
        result = retrieval_chain.invoke({'input': query_for_llm})
        processing_time = time.time() - start_time
        
        # Extract the answer and context
        answer = result.get('answer', '')
        context_docs = result.get('context', [])
        
        # Record this interaction
        self.add_to_history({
            "role": "user",
            "content": query_for_llm
        })
        self.add_to_history({
            "role": "assistant",
            "content": answer
        })
        
        # Prepare response with metadata
        response = {
            "answer": answer,
            "processing_time": processing_time,
            "retrieval_count": len(context_docs),
            "intent": intent,
            "subject": subject
        }
        
        # Add doc sources if available
        if context_docs:
            response["sources"] = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in context_docs
            ]
        
        return response

class GeminiValidatorAgent(Agent):
    """Agent using Gemini for validation and enhancement"""
    
    def __init__(self):
        super().__init__("Gemini Validator", "gemini_validator")
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance the Groq response using Gemini"""
        query = input_data.get("query", "")
        groq_response = input_data.get("groq_response", {})
        intent = input_data.get("intent", "concept_explanation")
        subject = input_data.get("subject")
        language = input_data.get("language", "en")
        
        # Extract the answer from Groq response
        groq_answer = groq_response.get("answer", "")
        
        # Craft a prompt for Gemini to validate and enhance
        language_instruction = ""
        if language.lower() == "fr":
            language_instruction = "Répondez en français."
        
        prompt = f"""
        {language_instruction}
        
        You are an expert ESI (École Supérieure d'Informatique) instructor specializing in computer science education.
        
        User query: {query}
        
        Base information provided by another AI system:
        {groq_answer}
        
        Your task is to:
        1. Validate the accuracy of the information provided
        2. Correct any factual errors or misconceptions
        3. Enhance the response with additional context, examples, or clarifications
        4. Improve the structure and clarity of the explanation
        5. Add code examples if relevant (especially for programming topics)
        6. Ensure the content is pedagogically appropriate for university students
        
        Detected Intent: {intent}
        Subject Area: {subject if subject else "General computer science"}
        
        Provide your enhanced response using appropriate academic language and structure.
        If you make significant corrections, briefly note what was corrected (but focus on providing the accurate information rather than critiquing).
        """
        
        # Process with Gemini
        start_time = time.time()
        try:
            response = self.model.generate_content(prompt)
            gemini_answer = response.text if hasattr(response, "text") else "".join(part.text for part in response.parts)
            processing_time = time.time() - start_time
            
            # Count approximate enhancements and corrections
            enhancements_count = 0
            corrections_count = 0
            
            # Simple heuristic: check if Gemini response is significantly different
            if gemini_answer and groq_answer:
                # If Gemini added significantly more content
                if len(gemini_answer) > len(groq_answer) * 1.3:
                    enhancements_count += 1
                
                # If Gemini contradicts Groq (simple check for phrases like "actually" or "however")
                correction_phrases = ["actually", "however", "correction", "instead", "rather", "more accurately"]
                for phrase in correction_phrases:
                    if phrase in gemini_answer.lower():
                        corrections_count += 1
            
            # Record this interaction
            self.add_to_history({
                "role": "system",
                "content": f"Validating response for query: {query}"
            })
            self.add_to_history({
                "role": "assistant",
                "content": gemini_answer
            })
            
            return {
                "enhanced_response": gemini_answer,
                "original_response": groq_answer,
                "processing_time": processing_time,
                "enhancements_count": enhancements_count,
                "corrections_count": corrections_count,
                "confidence": 0.9  # Placeholder
            }
            
        except Exception as e:
            # If Gemini fails, return original response
            return {
                "enhanced_response": groq_answer,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "enhancements_count": 0,
                "corrections_count": 0,
                "confidence": 0.5  # Lower confidence due to error
            }

class MultiAgentSystem:
    """Main class for the multi-agent system"""
    
    def __init__(self):
        # Initialize system components
        self.embeddings = None
        self.vector_store = None
        self.coordinator = None
        self.groq_agent = None
        self.gemini_agent = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the multi-agent system"""
        if self.is_initialized:
            return True
        
        try:
            # Initialize embeddings
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={
                    "batch_size": 64,
                    "normalize_embeddings": True
                }
            )
            
            # Load vector store
            save_directory = "./faiss_index"
            self.vector_store = FAISS.load_local(
                save_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Initialize agents
            self.coordinator = CoordinatorAgent()
            self.groq_agent = GroqRetrieverAgent(self.embeddings, self.vector_store)
            self.gemini_agent = GeminiValidatorAgent()
            
            # Register agents with coordinator
            self.coordinator.register_agents(self.groq_agent, self.gemini_agent)
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing multi-agent system: {str(e)}")
            traceback.print_exc()
            return False
    
    def query(self, query: str, language: str = "en") -> Dict[str, Any]:
        """Process a query through the multi-agent system"""
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                return {"error": "Failed to initialize multi-agent system"}
        
        try:
            result = self.coordinator.process({
                "query": query,
                "language": language
            })
            return result
        except Exception as e:
            error_message = f"Error processing query: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return {"error": error_message}

# Create singleton instance
_multi_agent_system = None

def get_multi_agent_system():
    """Get the singleton instance of the multi-agent system"""
    global _multi_agent_system
    if _multi_agent_system is None:
        _multi_agent_system = MultiAgentSystem()
        _multi_agent_system.initialize()
    return _multi_agent_system

def query_multi_agent_system(query: str, language: str = "en") -> Dict[str, Any]:
    """Query the multi-agent system with the specified query"""
    system = get_multi_agent_system()
    result = system.query(query, language)
    
    # Transform the result to match the format expected by the frontend
    if "error" in result:
        return {"error": result["error"]}
    
    final_result = {
        "enhanced_response": result["response"]["content"],
        "raw_response": result["groq_response"].get("answer", ""),
        "detected_intent": result["analysis"]["detected_intent"],
        "detected_subject": result["analysis"]["detected_subject"]
    }
    
    # Add sources/relevant documents if available
    if "sources" in result["groq_response"]:
        final_result["relevant_documents"] = result["groq_response"]["sources"]
    
    return final_result

# For testing
if __name__ == "__main__":

    
    st.title("Multi-Agent System Test")
    query = st.text_input("Enter a query")
    lang = st.selectbox("Language", ["en", "fr"])
    
    if st.button("Submit"):
        if query:
            with st.spinner("Processing..."):
                result = query_multi_agent_system(query, lang)
                st.write("### Response")
                st.write(result.get("enhanced_response", "No response"))
                
                with st.expander("Details"):
                    st.json({
                        "intent": result.get("detected_intent"),
                        "subject": result.get("detected_subject"),
                    })
        else:
            st.warning("Please enter a query")

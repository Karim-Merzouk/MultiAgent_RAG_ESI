import streamlit as st
import json
from rag_backend import (
    get_multi_agent_system,
    query_multi_agent_system,
    query_esi_rag
)

st.set_page_config(
    page_title="ESI Multi-Agent System Tester",
    layout="wide"
)

st.title("ESI Multi-Agent System Test Interface")
st.write("This interface tests the multi-agent system for ESI education assistance.")

# Initialize multi-agent system
if 'system_initialized' not in st.session_state:
    with st.spinner("Initializing multi-agent system..."):
        try:
            multi_agent = get_multi_agent_system()
            st.session_state.system_initialized = True
            st.success("✅ Multi-agent system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize multi-agent system: {e}")
            st.session_state.system_initialized = False

# Test interface
if st.session_state.system_initialized:
    st.subheader("Query Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Multi-Agent System")
        
        query_ma = st.text_area("Enter your query for multi-agent system:", 
                               height=100,
                               key="query_ma")
        language_ma = st.selectbox("Language:", ["en", "fr"], key="lang_ma")
        
        if st.button("Submit to Multi-Agent", key="submit_ma"):
            if query_ma:
                with st.spinner("Processing with multi-agent system..."):
                    try:
                        result = query_multi_agent_system(query_ma, language_ma)
                        
                        # Display results
                        st.subheader("Multi-Agent Result")
                        
                        # Display formatted response
                        st.markdown("#### Response:")
                        st.markdown(result.get("enhanced_response", "No response generated"))
                        
                        # Show analysis
                        with st.expander("Query Analysis"):
                            st.json({
                                "detected_intent": result.get("detected_intent", "unknown"),
                                "detected_subject": result.get("detected_subject", "unknown")
                            })
                        
                        # Show raw response
                        with st.expander("Raw Response"):
                            st.text(result.get("raw_response", "No raw response available"))
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a query.")
    
    with col2:
        st.write("### Standard RAG System")
        
        query_rag = st.text_area("Enter your query for standard RAG:", 
                                height=100,
                                key="query_rag")
        language_rag = st.selectbox("Language:", ["en", "fr"], key="lang_rag")
        
        if st.button("Submit to Standard RAG", key="submit_rag"):
            if query_rag:
                with st.spinner("Processing with standard RAG..."):
                    try:
                        result = query_esi_rag(query_rag, system_mode="auto", language=language_rag)
                        
                        # Display results
                        st.subheader("Standard RAG Result")
                        
                        # Display formatted response
                        st.markdown("#### Response:")
                        st.markdown(result.get("enhanced_response", "No response generated"))
                        
                        # Show analysis
                        with st.expander("Query Analysis"):
                            st.json({
                                "detected_intent": result.get("detected_intent", "unknown"),
                                "detected_subject": result.get("detected_subject", "unknown")
                            })
                        
                        # Show raw response
                        with st.expander("Raw Response"):
                            st.text(result.get("raw_response", "No raw response available"))
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a query.")
    
    # Comparison feature
    st.markdown("---")
    st.subheader("Comparative Analysis")
    
    compare_query = st.text_area("Enter a query to compare both systems:", 
                               height=100,
                               key="query_compare")
    compare_language = st.selectbox("Language:", ["en", "fr"], key="lang_compare")
    
    if st.button("Run Comparison", key="submit_compare"):
        if compare_query:
            col1, col2 = st.columns(2)
            
            with st.spinner("Running comparison..."):
                try:
                    # Run both systems in parallel
                    result_ma = query_multi_agent_system(compare_query, compare_language)
                    result_rag = query_esi_rag(compare_query, system_mode="auto", language=compare_language)
                    
                    # Display results side by side
                    with col1:
                        st.subheader("Multi-Agent System")
                        st.markdown(result_ma.get("enhanced_response", "No response generated"))
                        
                        # Show processing metadata
                        with st.expander("Processing Details"):
                            if isinstance(result_ma, dict) and "metrics" in result_ma:
                                st.json(result_ma["metrics"])
                            else:
                                st.write("No detailed metrics available")
                    
                    with col2:
                        st.subheader("Standard RAG")
                        st.markdown(result_rag.get("enhanced_response", "No response generated"))
                
                except Exception as e:
                    st.error(f"Comparison error: {e}")
        else:
            st.warning("Please enter a query for comparison.")
else:
    st.warning("Multi-agent system is not initialized. Please reload the page or check the backend.")

import streamlit as st
import subprocess
import sys
import os

def main():
    """Launch the Streamlit interface for the Multi-agent system."""
    print("Starting ESI Multi-Agent System...")
    
    # Check if required modules are installed
    try:
        import streamlit
        import tryllm
    except ImportError as e:
        print(f"Error: Missing required module. {e}")
        print("Please install required packages with: pip install streamlit langchain langchain_groq google-generativeai")
        return

    # Get the path to this script
    script_path = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the interface script
    interface_path = os.path.join(script_path, "streamlit_interface.py")
    
    # Launch Streamlit
    print(f"Launching Streamlit interface: {interface_path}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", interface_path])
    except Exception as e:
        print(f"Error launching Streamlit: {e}")
        print("Try running manually with: streamlit run streamlit_interface.py")

if __name__ == "__main__":
    main()

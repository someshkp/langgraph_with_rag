import streamlit as st
import os
from dotenv import load_dotenv
from agent import agent
from ingest import main as run_ingestion
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="LangGraph RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e0e0e0;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #2d2d2d;
    }
    .stChatMessage[data-testimonial="user"] {
        background-color: #1e222d;
    }
    .stChatMessage[data-testimonial="assistant"] {
        background-color: #262730;
    }
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #238636;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("---")
    
    if st.button("🔄 Re-index Documents"):
        with st.spinner("Indexing documents... This may take a moment."):
            try:
                run_ingestion()
                st.success("Indexing complete!")
            except Exception as e:
                st.error(f"Error during indexing: {e}")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This assistant uses **LangGraph** for orchestration and **FAISS** for RAG.
    
    Tools available:
    - 🔍 **search_docs**: Search the knowledge base.
    - ➕ **add**: Add numbers.
    - ✖️ **multiply**: Multiply numbers.
    - ➗ **Divide**: Divide numbers.
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Header
st.title("🤖 LangGraph RAG Assistant")
st.markdown("Welcome! Ask me anything about AI/ML or use my math tools.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare the input for LangGraph
                # We send the entire message history to maintain context
                input_state = {
                    "messages": st.session_state.messages,
                    "llm_calls": 0 # Initialize llm_calls if needed
                }
                
                # Stream the response from the agent
                # Note: agent.invoke or agent.stream can be used. 
                # For simplicity in this UI, we'll use invoke.
                final_state = agent.invoke(input_state)
                
                # Get the last message from the updated state
                # The state contains the full message list (Annotated operator.add)
                response_message = final_state["messages"][-1]
                
                if isinstance(response_message, AIMessage):
                    full_response = response_message.content
                    st.markdown(full_response)
                    # Add assistant message to history
                    st.session_state.messages.append(response_message)
                else:
                    # In case the last message is a ToolMessage (though agent should end with AIMessage)
                    st.error("Unexpected response format.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

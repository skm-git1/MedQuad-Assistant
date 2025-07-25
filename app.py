import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import spacy
import re
from typing import List, Dict, Tuple
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Medical Q&A Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
        color: #28a745;
    }
    .entity-tag {
        background-color: #007bff;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

class MedicalQAStreamlitApp:
    def __init__(self):
        self.qa_system = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'qa_system' not in st.session_state:
            st.session_state.qa_system = None
    
    def load_qa_system(self, model_path: str, gemini_api_key: str):
        """Load the trained Q&A system"""
        try:
            with st.spinner("Loading medical Q&A model..."):
                # Load model data
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Recreate the QA system
                from train_model import MedicalQASystem
                qa_system = MedicalQASystem(gemini_api_key)
                qa_system.qa_data = model_data['qa_data']
                qa_system.embeddings = model_data['embeddings']
                qa_system.vectorizer = model_data['vectorizer']
                qa_system.tfidf_matrix = model_data['tfidf_matrix']
                
                return qa_system
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def display_medical_entities(self, entities: Dict[str, List[str]]):
        """Display extracted medical entities with colored tags"""
        if any(entities.values()):
            st.markdown("**🔍 Detected Medical Entities:**")
            
            cols = st.columns(4)
            entity_types = ['symptoms', 'diseases', 'treatments', 'body_parts']
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            for i, (entity_type, entities_list) in enumerate(zip(entity_types, [entities[t] for t in entity_types])):
                with cols[i]:
                    if entities_list:
                        st.markdown(f"**{entity_type.title()}:**")
                        for entity in entities_list:
                            st.markdown(f'<span class="entity-tag" style="background-color: {colors[i]}">{entity}</span>', 
                                      unsafe_allow_html=True)
    
    def display_confidence_meter(self, confidence: float):
        """Display confidence score as a meter"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Answer Confidence (%)"},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def display_similar_questions(self, similar_questions: List[Tuple[str, float]]):
        """Display similar questions from the dataset"""
        if similar_questions:
            st.markdown("**📚 Related Questions from Medical Database:**")
            for i, (question, similarity) in enumerate(similar_questions):
                with st.expander(f"Similar Question {i+1} (Similarity: {similarity:.3f})"):
                    st.write(question)
    
    def format_answer(self, answer: str) -> str:
        """Format the answer with better structure"""
        # Split into paragraphs and add bullet points where appropriate
        paragraphs = answer.split('\n\n')
        formatted_paragraphs = []
        
        for para in paragraphs:
            if para.strip():
                # If paragraph contains numbered points, format them
                if re.search(r'\d+\.', para):
                    lines = para.split('\n')
                    formatted_lines = []
                    for line in lines:
                        if re.match(r'^\d+\.', line.strip()):
                            formatted_lines.append(f"**{line.strip()}**")
                        else:
                            formatted_lines.append(line.strip())
                    formatted_paragraphs.append('\n'.join(formatted_lines))
                else:
                    formatted_paragraphs.append(para.strip())
        
        return '\n\n'.join(formatted_paragraphs)
    
    def run_app(self):
        """Main Streamlit app"""
        # Header
        st.markdown('<h1 class="main-header">🏥 Medical Q&A Assistant</h1>', unsafe_allow_html=True)
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # API Key input
            gemini_api_key = st.text_input(
                "Enter Gemini API Key:",
                type="password",
                help="Get your API key from Google AI Studio"
            )
            
            # Model file upload/selection
            model_file = st.file_uploader(
                "Upload trained model file (medical_qa_model.pkl):",
                type=['pkl'],
                help="Upload the trained model file created by train_model.py"
            )
            
            # Load model button
            if st.button("Load Model") and gemini_api_key and model_file:
                # Save uploaded file temporarily
                with open("temp_model.pkl", "wb") as f:
                    f.write(model_file.getbuffer())
                
                qa_system = self.load_qa_system("temp_model.pkl", gemini_api_key)
                if qa_system:
                    st.session_state.qa_system = qa_system
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                    os.remove("temp_model.pkl")  # Clean up
                else:
                    st.error("❌ Failed to load model")
            
            # Model status
            if st.session_state.model_loaded:
                st.success("🟢 Model Status: Ready")
                if st.session_state.qa_system:
                    st.info(f"📊 Dataset: {len(st.session_state.qa_system.qa_data)} Q&A pairs")
            else:
                st.warning("🟡 Model Status: Not loaded")
            
            # Chat history management
            st.header("💬 Chat History")
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared!")
            
            if st.session_state.chat_history:
                st.info(f"Total questions asked: {len(st.session_state.chat_history)}")
        
        # Main content area
        if not st.session_state.model_loaded:
            st.markdown("""
            <div class="warning-box">
                <h3>🚀 Getting Started</h3>
                <ol>
                    <li>Enter your Gemini API key in the sidebar</li>
                    <li>Upload your trained model file (medical_qa_model.pkl)</li>
                    <li>Click "Load Model" to initialize the system</li>
                    <li>Start asking medical questions!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample questions
            st.markdown("### 💡 Sample Questions You Can Ask:")
            sample_questions = [
                "What are the symptoms of diabetes?",
                "How to treat high blood pressure?",
                "What causes chest pain?",
                "What are the side effects of aspirin?",
                "How to prevent heart disease?"
            ]
            
            for question in sample_questions:
                st.markdown(f"• {question}")
            
        else:
            # Question input
            st.markdown("### 🤔 Ask Your Medical Question:")
            
            # Input methods
            input_method = st.radio(
                "Choose input method:",
                ["Type question", "Select from examples"],
                horizontal=True
            )
            
            if input_method == "Type question":
                user_question = st.text_area(
                    "Enter your medical question:",
                    height=100,
                    placeholder="e.g., What are the symptoms of diabetes?"
                )
            else:
                example_questions = [
                    "What are the symptoms of diabetes?",
                    "How to treat high blood pressure?",
                    "What causes chest pain?",
                    "What are the side effects of aspirin?",
                    "How to prevent heart disease?",
                    "What is the difference between Type 1 and Type 2 diabetes?",
                    "How to manage anxiety naturally?",
                    "What foods should I avoid with high cholesterol?"
                ]
                user_question = st.selectbox("Select an example question:", example_questions)
            
            # Submit button
            if st.button("🔍 Get Answer", type="primary") and user_question.strip():
                with st.spinner("Searching medical database and generating answer..."):
                    # Get answer from the system
                    result = st.session_state.qa_system.answer_question(user_question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'entities': result['entities'],
                        'similar_questions': result['similar_questions'],
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                
                # Display results
                st.markdown("---")
                
                # Question display
                st.markdown(f'<div class="question-box"><strong>❓ Your Question:</strong><br>{user_question}</div>', 
                          unsafe_allow_html=True)
                
                # Create columns for answer and confidence
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Answer display
                    formatted_answer = self.format_answer(result['answer'])
                    st.markdown(f'<div class="answer-box"><strong>💡 Answer:</strong><br>{formatted_answer}</div>', 
                              unsafe_allow_html=True)
                
                with col2:
                    # Confidence score
                    self.display_confidence_meter(result['confidence'])
                
                # Medical entities
                self.display_medical_entities(result['entities'])
                
                # Similar questions
                self.display_similar_questions(result['similar_questions'])
                
                # Medical disclaimer
                st.markdown("""
                <div class="warning-box">
                    <strong>⚠️ Medical Disclaimer:</strong> This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for personalized medical guidance.
                </div>
                """, unsafe_allow_html=True)
        
        # Chat History Display
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📝 Recent Chat History")
            
            # Show last 5 conversations
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                with st.expander(f"Q{len(st.session_state.chat_history)-i}: {chat['question'][:100]}..." if len(chat['question']) > 100 else f"Q{len(st.session_state.chat_history)-i}: {chat['question']}"):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer'][:300]}..." if len(chat['answer']) > 300 else chat['answer'])
                    st.markdown(f"**Confidence:** {chat['confidence']:.3f}")
                    st.markdown(f"**Timestamp:** {chat['timestamp']}")

def main():
    """Main function to run the Streamlit app"""
    app = MedicalQAStreamlitApp()
    app.run_app()

if __name__ == "__main__":
    main()
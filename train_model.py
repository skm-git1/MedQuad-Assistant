import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import spacy
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

class MedicalQASystem:
    def __init__(self, gemini_api_key: str):
        """
        Initialize the Medical Q&A System
        
        Args:
            gemini_api_key (str): Your Gemini API key
        """
        self.gemini_api_key = gemini_api_key
        self.setup_gemini()
        self.load_models()
        self.vectorizer = None
        self.embeddings = None
        self.qa_data = None
        
    def setup_gemini(self):
        """Setup Gemini API"""
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def load_models(self):
        """Load required NLP models"""
        print("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model 'en_core_web_sm' not found. Please install it using:")
            print("python -m spacy download en_core_web_sm")
            # Use a basic tokenizer as fallback
            self.nlp = None
    
    def load_dataset(self, csv_path: str):
        """
        Load and preprocess the QA dataset
        
        Args:
            csv_path (str): Path to the CSV file containing Q&A data
        """
        print(f"Loading dataset from {csv_path}...")
        self.qa_data = pd.read_csv(csv_path)
        
        # Basic data cleaning
        self.qa_data = self.qa_data.dropna(subset=['Question', 'Answer'])
        self.qa_data['Question'] = self.qa_data['Question'].astype(str)
        self.qa_data['Answer'] = self.qa_data['Answer'].astype(str)
        
        # Clean text
        self.qa_data['Question_clean'] = self.qa_data['Question'].apply(self.clean_text)
        self.qa_data['Answer_clean'] = self.qa_data['Answer'].apply(self.clean_text)
        
        print(f"Dataset loaded with {len(self.qa_data)} Q&A pairs")
        
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.\,\(\)]', '', text)
        return text.lower()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from text using spaCy and custom patterns
        
        Args:
            text (str): Input text
            
        Returns:
            Dict containing extracted entities
        """
        entities = {
            'symptoms': [],
            'diseases': [],
            'treatments': [],
            'body_parts': []
        }
        
        if self.nlp is None:
            return entities
            
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'SYMPTOM']:
                entities['diseases'].append(ent.text.lower())
        
        # Custom medical term patterns
        medical_keywords = {
            'symptoms': ['pain', 'ache', 'fever', 'headache', 'nausea', 'vomiting', 
                        'dizziness', 'fatigue', 'weakness', 'swelling', 'rash',
                        'cough', 'shortness of breath', 'chest pain'],
            'diseases': ['diabetes', 'hypertension', 'cancer', 'heart disease',
                        'stroke', 'arthritis', 'asthma', 'pneumonia', 'flu',
                        'covid', 'depression', 'anxiety'],
            'treatments': ['medication', 'surgery', 'therapy', 'treatment',
                          'antibiotics', 'vaccine', 'exercise', 'diet'],
            'body_parts': ['heart', 'lung', 'brain', 'liver', 'kidney',
                          'stomach', 'chest', 'head', 'back', 'joint']
        }
        
        text_lower = text.lower()
        for category, keywords in medical_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities[category].append(keyword)
        
        # Remove duplicates
        for category in entities:
            entities[category] = list(set(entities[category]))
            
        return entities
    
    def create_embeddings(self):
        """Create embeddings for questions using sentence transformers"""
        print("Creating embeddings for questions...")
        questions = self.qa_data['Question_clean'].tolist()
        
        # Create sentence embeddings
        self.embeddings = self.sentence_model.encode(questions)
        
        # Also create TF-IDF vectors as backup
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)
        
        print("Embeddings created successfully!")
    
    def find_similar_questions(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str, str]]:
        """
        Find similar questions using semantic similarity
        
        Args:
            query (str): User query
            top_k (int): Number of similar questions to return
            
        Returns:
            List of tuples (index, similarity_score, question, answer)
        """
        query_clean = self.clean_text(query)
        
        # Create embedding for query
        query_embedding = self.sentence_model.encode([query_clean])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k similar questions
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                idx,
                similarities[idx],
                self.qa_data.iloc[idx]['Question'],
                self.qa_data.iloc[idx]['Answer']
            ))
        
        return results
    
    def generate_enhanced_answer(self, query: str, context_answers: List[str]) -> str:
        """
        Generate an enhanced answer using Gemini with context from similar Q&As
        
        Args:
            query (str): User query
            context_answers (List[str]): Relevant answers from dataset
            
        Returns:
            Enhanced answer generated by Gemini
        """
        # Extract medical entities from query
        entities = self.extract_medical_entities(query)
        
        # Create context from similar answers
        context = "\n".join([f"Context {i+1}: {answer}" for i, answer in enumerate(context_answers[:3])])
        
        # Create prompt for Gemini
        prompt = f"""
        You are a medical AI assistant. Based on the following context from medical literature and the user's question, provide a comprehensive and accurate answer.

        Medical entities identified in question:
        - Symptoms: {', '.join(entities['symptoms']) if entities['symptoms'] else 'None'}
        - Diseases: {', '.join(entities['diseases']) if entities['diseases'] else 'None'}
        - Treatments: {', '.join(entities['treatments']) if entities['treatments'] else 'None'}
        - Body parts: {', '.join(entities['body_parts']) if entities['body_parts'] else 'None'}

        Context from medical database:
        {context}

        User Question: {query}

        Please provide a comprehensive answer that:
        1. Addresses the user's specific question
        2. Uses the provided context appropriately
        3. Includes relevant medical information
        4. Maintains accuracy and clarity
        5. Includes appropriate medical disclaimers when necessary

        Important: Always remind users to consult healthcare professionals for personalized medical advice.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating enhanced answer: {str(e)}")
            # Fallback to the most similar answer from dataset
            if context_answers:
                return context_answers[0] + "\n\nNote: Please consult a healthcare professional for personalized medical advice."
            return "I apologize, but I couldn't generate a proper response. Please consult a healthcare professional for medical advice."
    
    def answer_question(self, query: str) -> Dict:
        """
        Main method to answer a medical question
        
        Args:
            query (str): User's medical question
            
        Returns:
            Dictionary containing answer and metadata
        """
        # Find similar questions
        similar_questions = self.find_similar_questions(query, top_k=5)
        
        if not similar_questions or similar_questions[0][1] < 0.3:  # Low similarity threshold
            return {
                'answer': "I don't have enough information to answer this question accurately. Please consult a healthcare professional for medical advice.",
                'confidence': 0.0,
                'similar_questions': [],
                'entities': self.extract_medical_entities(query)
            }
        
        # Extract answers from similar questions
        context_answers = [item[3] for item in similar_questions]
        
        # Generate enhanced answer using Gemini
        enhanced_answer = self.generate_enhanced_answer(query, context_answers)
        
        return {
            'answer': enhanced_answer,
            'confidence': similar_questions[0][1],
            'similar_questions': [(q[2], q[1]) for q in similar_questions[:3]],
            'entities': self.extract_medical_entities(query)
        }
    
    def save_model(self, model_path: str = 'medical_qa_model.pkl'):
        """Save the trained model components"""
        print(f"Saving model to {model_path}...")
        
        model_data = {
            'qa_data': self.qa_data,
            'embeddings': self.embeddings,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix
            # 'gemini_api_key': self.gemini_api_key
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved successfully!")
    
    def load_model(self, model_path: str = 'medical_qa_model.pkl'):
        """Load a pre-trained model"""
        print(f"Loading model from {model_path}...")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.qa_data = model_data['qa_data']
        self.embeddings = model_data['embeddings']
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        
        print("Model loaded successfully!")

def main():
    # Configuration
    GEMINI_API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]  # Replace with your actual API key
    CSV_FILE_PATH = "qa_dataset_cleaned.csv"  # Path to your dataset
    
    # Initialize the system
    print("Initializing Medical Q&A System...")
    qa_system = MedicalQASystem(GEMINI_API_KEY)
    
    # Load and process dataset
    qa_system.load_dataset(CSV_FILE_PATH)
    
    # Create embeddings
    qa_system.create_embeddings()
    
    # Save the model
    qa_system.save_model()
    
    # Test the system
    print("\nTesting the system with sample questions...")
    test_questions = [
        "What are the symptoms of diabetes?",
        "How to treat high blood pressure?",
        "What causes chest pain?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        result = qa_system.answer_question(question)
        print(f"Answer: {result['answer'][:]}...")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Entities: {result['entities']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
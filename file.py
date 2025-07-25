## i have saved the trained model in a pickle file with name medical_qa_model.pkl
## now i want to load this model and use it to answer questions
import pickle
# Load the trained model from the pickle file
with open('medical_qa_model.pkl', 'rb') as file:    
    model = pickle.load(file)

# Now you can use the loaded model to answer questions
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

def load_qa_pairs_from_csv(csv_file):
    """Load question-answer pairs from a CSV file."""
    qa_pairs = []
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            qa_pairs.append(row)
    return qa_pairs

def answer_question(question, qa_pairs):
    """Answer a question based on the loaded QA pairs."""
    for pair in qa_pairs:
        if pair['Question'].lower() == question.lower():
            return pair['Answer']
    return "Sorry, I don't have an answer for that."


def main():
    # Define the path to the CSV file
    csv_file = Path('dataset/qa_dataset.csv')
    
    # Load QA pairs from the CSV file
    qa_pairs = load_qa_pairs_from_csv(csv_file)
    
    # Example questions to answer
    questions = [
        "What are the symptoms of diabetes?",
        "How does AIDS affect a person?",
        "What are the primary causes of pneumonia?"
    ]
    
    # Answer each question using the loaded model
    for question in questions:
        answer = answer_question(question, qa_pairs)
        print(f"Question: {question}\nAnswer: {answer}\n")

if __name__ == "__main__":
    main()
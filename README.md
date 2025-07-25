## MedQuad Assistant
A Medical Q&A Chatbot trained on the MedQuAD Dataset[https://github.com/abachaa/MedQuAD], implemented using a retrieval mechanism to find relevant answers. 
Basic medical entity recognition (e.g., symptoms, diseases, treatments). 
User interface using Streamlit for asking medical questions

Dataset - made from the MedQuAD Dataset using the xml_to_csv.py script - qa_dataset_cleaned.csv
Model training script - train_model.py
Streamlit app script - appp.py

Model file - medical_qa_model.pkl

To run locally - 
1. Clone the repo
2. Run the train_model.py
   ```python
     python train_model.py
   ```
3. After that run the appp.py file
   ```python
     streamlit run appp.py
   ```

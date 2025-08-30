import os
import pickle
import numpy as np
from deepface import DeepFace
import streamlit as st
from employee import load_employee_data

def save_embeddings_file():
    """Save embeddings to separate file for faster loading during video processing"""
    employees_data = load_employee_data()
    embeddings_data = {
        'ids': [],
        'names': [],
        'embeddings': []
    }
    
    for emp_id, emp_data in employees_data.items():
        # Add all embeddings for each employee
        for embedding in emp_data.get('embeddings', []):
            if embedding is not None:
                embeddings_data['ids'].append(emp_id)        # Save ID
                embeddings_data['names'].append(emp_data['name'])
                embeddings_data['embeddings'].append(embedding)
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_data, f)



def load_embeddings_for_recognition():
    """Load pre-computed embeddings for fast recognition"""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return {'ids': [], 'names': [], 'embeddings': []}




def compute_embedding(image_path):
    """Compute embedding for a single image using DeepFace"""
    try:
        # Generate embedding using DeepFace
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name='VGG-Face',
            enforce_detection=False
        )
        return embedding[0]['embedding']  # Return the embedding vector
    except Exception as e:
        st.error(f"Error computing embedding: {str(e)}")
        return None

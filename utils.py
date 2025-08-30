import numpy as np
import os
import pickle


EMBEDDINGS_FILE = "employee_embeddings.pkl"
EMPLOYEE_DATA_FILE = "employee_data.pkl"

def cosine_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)


def load_employee_data():
    """Load all employee data from file"""
    if os.path.exists(EMPLOYEE_DATA_FILE):
        with open(EMPLOYEE_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

import os
import pickle
from embeddings import save_embeddings_file
from utils import load_employee_data, EMPLOYEE_DATA_FILE





def generate_employee_id(employees_data):
    if not employees_data:
        return 1
    return max(int(emp_id) for emp_id in employees_data.keys()) + 1



def save_employee_data(name, photo_path, embedding):
    """Save employee data and embedding to files"""
    # Load existing data
    employees_data = load_employee_data()
    
    # Check if employee already exists
    existing_emp_id = None
    for emp_id, emp_data in employees_data.items():
        if emp_data['name'].lower() == name.lower():
            existing_emp_id = emp_id
            break
    
    if existing_emp_id:
        # Add new photo and embedding to existing employee
        employees_data[existing_emp_id]['photo_paths'].append(photo_path)
        employees_data[existing_emp_id]['embeddings'].append(embedding)
    else:
        # Create new employee
        employee_id = generate_employee_id(employees_data)
        employees_data[employee_id] = {
            'name': name,
            'photo_paths': [photo_path],
            'embeddings': [embedding]
        }
    
    # Save to file
    with open(EMPLOYEE_DATA_FILE, 'wb') as f:
        pickle.dump(employees_data, f)
    
    # Update embeddings file for fast loading
    save_embeddings_file()





def get_all_employees():
    """Get all employees from file"""
    employees_data = load_employee_data()
    employees_list = []
    for emp_id, emp_data in employees_data.items():
        photo_paths = emp_data.get('photo_paths', [])
        photo_count = len(photo_paths)
        main_photo = photo_paths[0] if photo_paths else None
        employees_list.append((emp_id, emp_data['name'], main_photo, photo_count, photo_paths))
    return employees_list



def delete_employee(employee_id):
    """Delete employee from files"""
    employees_data = load_employee_data()
    if employee_id in employees_data:
        # Remove photo files if they exist
        photo_paths = employees_data[employee_id].get('photo_paths', [])
        for photo_path in photo_paths:
            if photo_path and os.path.exists(photo_path):
                os.remove(photo_path)
        
        # Remove from data
        del employees_data[employee_id]
        
        # Save updated data
        with open(EMPLOYEE_DATA_FILE, 'wb') as f:
            pickle.dump(employees_data, f)

        save_embeddings_file()

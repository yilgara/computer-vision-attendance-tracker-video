import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, date
import tempfile
from PIL import Image

import time

# File-based storage for embeddings
EMBEDDINGS_FILE = "employee_embeddings.pkl"
EMPLOYEE_DATA_FILE = "employee_data.pkl"

def save_employee_data(name, photo_path, encoding):
    """Save employee data and embedding to files"""
    # Load existing data
    employees_data = load_employee_data()
    
    # Add new employee
    employee_id = len(employees_data) + 1
    employees_data[employee_id] = {
        'name': name,
        'photo_path': photo_path,
        'embedding': encoding
    }
    
    # Save to file
    with open(EMPLOYEE_DATA_FILE, 'wb') as f:
        pickle.dump(employees_data, f)
    
    # Update embeddings file
    save_embeddings_file()

def load_employee_data():
    """Load all employee data from file"""
    if os.path.exists(EMPLOYEE_DATA_FILE):
        with open(EMPLOYEE_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def save_embeddings_file():
    """Save embeddings to separate file for faster loading"""
    employees_data = load_employee_data()
    embeddings_data = {
        'names': [],
        'encodings': []
    }
    
    for emp_id, emp_data in employees_data.items():
        embeddings_data['names'].append(emp_data['name'])
        embeddings_data['encodings'].append(emp_data['embedding'])
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_data, f)

def load_employee_embeddings():
    """Load employee embeddings from file"""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings_data = pickle.load(f)
            return embeddings_data['encodings'], embeddings_data['names']
    return [], []

def get_all_employees():
    """Get all employees from file"""
    employees_data = load_employee_data()
    employees_list = []
    for emp_id, emp_data in employees_data.items():
        employees_list.append((emp_id, emp_data['name'], emp_data['photo_path']))
    return employees_list

def delete_employee(employee_id):
    """Delete employee from files"""
    employees_data = load_employee_data()
    if employee_id in employees_data:
        # Remove photo file if it exists
        photo_path = employees_data[employee_id]['photo_path']
        if os.path.exists(photo_path):
            os.remove(photo_path)
        
        # Remove from data
        del employees_data[employee_id]
        
        # Save updated data
        with open(EMPLOYEE_DATA_FILE, 'wb') as f:
            pickle.dump(employees_data, f)
        
        # Update embeddings file
        save_embeddings_file()

def create_attendance_log(date_str, employee_name, entry_time, exit_time=None):
    """Create or update attendance log in Excel"""
    filename = f"attendance_{date_str}.xlsx"
    
    # Check if file exists
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=['Employee', 'Date', 'Entry Time', 'Exit Time', 'Total Hours'])
    
    # Check if employee already has an entry for today
    existing_entry = df[(df['Employee'] == employee_name) & (df['Date'] == date_str)]
    
    if not existing_entry.empty:
        # Update exit time if it's an exit
        if exit_time:
            idx = existing_entry.index[0]
            df.at[idx, 'Exit Time'] = exit_time
            # Calculate total hours
            if pd.notna(df.at[idx, 'Entry Time']):
                entry_dt = pd.to_datetime(f"{date_str} {df.at[idx, 'Entry Time']}")
                exit_dt = pd.to_datetime(f"{date_str} {exit_time}")
                total_hours = (exit_dt - entry_dt).total_seconds() / 3600
                df.at[idx, 'Total Hours'] = round(total_hours, 2)
    else:
        # Create new entry
        new_row = {
            'Employee': employee_name,
            'Date': date_str,
            'Entry Time': entry_time,
            'Exit Time': exit_time,
            'Total Hours': None
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    df.to_excel(filename, index=False)
    return filename

def process_video_for_attendance(video_path, camera_type="entry"):
    """Process video file and detect faces for attendance"""
    known_encodings, known_names = load_employee_embeddings()
    
    if not known_encodings:
        st.error("No employee embeddings found. Please add employees first.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    detected_employees = set()
    attendance_logs = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process every 30th frame to speed up processing
    frame_skip = 30
    current_frame = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.read()[0]:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_skip == 0:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            # Find faces and encodings
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]
                    if name not in detected_employees:
                        detected_employees.add(name)
                        
                        # Calculate timestamp based on frame position
                        timestamp_seconds = current_frame / fps
                        hours = int(timestamp_seconds // 3600)
                        minutes = int((timestamp_seconds % 3600) // 60)
                        seconds = int(timestamp_seconds % 60)
                        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        attendance_logs.append({
                            'name': name,
                            'time': timestamp,
                            'type': camera_type
                        })
        
        current_frame += 1
        progress = current_frame / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {current_frame}/{total_frames}")
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return attendance_logs

def main():
    st.set_page_config(page_title="Employee Attendance System", layout="wide")
    st.title("🏢 Employee Attendance System")
    st.markdown("---")
    
    # Create necessary directories
    os.makedirs("employee_photos", exist_ok=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Employee Management", "Process Attendance", "View Records"])
    
    if page == "Employee Management":
        st.header("👥 Employee Management")
        
        tab1, tab2 = st.tabs(["Add Employee", "Manage Employees"])
        
        with tab1:
            st.subheader("Add New Employee")
            
            employee_name = st.text_input("Employee Name")
            uploaded_photo = st.file_uploader("Upload Employee Photo", 
                                            type=['jpg', 'jpeg', 'png'])
            
            if st.button("Add Employee") and employee_name and uploaded_photo:
                # Save uploaded photo
                photo_dir = "employee_photos"
                os.makedirs(photo_dir, exist_ok=True)
                photo_path = os.path.join(photo_dir, f"{employee_name}_{uploaded_photo.name}")
                
                with open(photo_path, "wb") as f:
                    f.write(uploaded_photo.getbuffer())
                
                # Load and process the image
                image = face_recognition.load_image_file(photo_path)
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    face_encoding = face_encodings[0]
                    save_employee_data(employee_name, photo_path, face_encoding)
                    st.success(f"Employee {employee_name} added successfully!")
                    st.rerun()
                else:
                    st.error("No face detected in the uploaded photo. Please upload a clear photo.")
                    os.remove(photo_path)
        
        with tab2:
            st.subheader("Manage Employees")
            
            employees = get_all_employees()
            if employees:
                for emp_id, name, photo_path in employees:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"**{name}**")
                    with col2:
                        if os.path.exists(photo_path):
                            st.image(photo_path, width=100)
                    with col3:
                        if st.button("Delete", key=f"del_{emp_id}"):
                            delete_employee(emp_id)
                            st.success(f"Employee {name} deleted!")
                            st.rerun()
            else:
                st.info("No employees found. Please add employees first.")
    
    elif page == "Process Attendance":
        st.header("📹 Process Attendance Videos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entry Camera")
            entry_video = st.file_uploader("Upload Entry Video", 
                                         type=['mp4', 'avi', 'mov'], 
                                         key="entry")
            
            if entry_video and st.button("Process Entry Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(entry_video.getbuffer())
                    tmp_path = tmp.name
                
                st.info("Processing entry video... This may take a few minutes.")
                attendance_logs = process_video_for_attendance(tmp_path, "entry")
                os.unlink(tmp_path)
                
                if attendance_logs:
                    today = date.today().strftime("%Y-%m-%d")
                    for log in attendance_logs:
                        filename = create_attendance_log(today, log['name'], log['time'])
                    
                    st.success(f"Entry processing complete! {len(attendance_logs)} employees detected.")
                    for log in attendance_logs:
                        st.write(f"✅ {log['name']} entered at {log['time']}")
                else:
                    st.warning("No employees detected in the entry video.")
        
        with col2:
            st.subheader("Exit Camera")
            exit_video = st.file_uploader("Upload Exit Video", 
                                        type=['mp4', 'avi', 'mov'], 
                                        key="exit")
            
            if exit_video and st.button("Process Exit Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(exit_video.getbuffer())
                    tmp_path = tmp.name
                
                st.info("Processing exit video... This may take a few minutes.")
                attendance_logs = process_video_for_attendance(tmp_path, "exit")
                os.unlink(tmp_path)
                
                if attendance_logs:
                    today = date.today().strftime("%Y-%m-%d")
                    for log in attendance_logs:
                        filename = create_attendance_log(today, log['name'], None, log['time'])
                    
                    st.success(f"Exit processing complete! {len(attendance_logs)} employees detected.")
                    for log in attendance_logs:
                        st.write(f"🚪 {log['name']} exited at {log['time']}")
                else:
                    st.warning("No employees detected in the exit video.")
    
    elif page == "View Records":
        st.header("📊 Attendance Records")
        
        # List available attendance files
        attendance_files = [f for f in os.listdir('.') if f.startswith('attendance_') and f.endswith('.xlsx')]
        
        if attendance_files:
            selected_file = st.selectbox("Select Date", attendance_files)
            
            if selected_file:
                df = pd.read_excel(selected_file)
                st.dataframe(df, use_container_width=True)
                
                # Download button
                with open(selected_file, 'rb') as f:
                    st.download_button(
                        label=f"Download {selected_file}",
                        data=f.read(),
                        file_name=selected_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Summary statistics
                st.subheader("Daily Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Employees", len(df))
                with col2:
                    present = len(df[df['Entry Time'].notna()])
                    st.metric("Present", present)
                with col3:
                    avg_hours = df['Total Hours'].mean() if 'Total Hours' in df.columns else 0
                    st.metric("Avg Hours", f"{avg_hours:.1f}" if not pd.isna(avg_hours) else "N/A")
        else:
            st.info("No attendance records found. Process some videos first.")

if __name__ == "__main__":
    main()

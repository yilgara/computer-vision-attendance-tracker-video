import streamlit as st
import cv2
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, date
import tempfile
from PIL import Image
import time
from deepface import DeepFace
import warnings
warnings.filterwarnings('ignore')

# File-based storage for embeddings
EMBEDDINGS_FILE = "employee_embeddings.pkl"
EMPLOYEE_DATA_FILE = "employee_data.pkl"


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
    return {'names': [], 'embeddings': []}

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

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm_a = np.linalg.norm(embedding1)
    norm_b = np.linalg.norm(embedding2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def load_employee_data():
    """Load all employee data from file"""
    if os.path.exists(EMPLOYEE_DATA_FILE):
        with open(EMPLOYEE_DATA_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

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

def recognize_face_in_frame(frame, known_embeddings, known_names, threshold=0.6):
    """Recognize faces in frame using pre-computed embeddings (MUCH FASTER!)"""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute embedding for the frame using DeepFace in-memory
        embeddings_result = DeepFace.represent(
            img_path=None,  # Not using a path
            img=rgb_frame,  # Pass the NumPy array directly
            model_name='VGG-Face',
            enforce_detection=False
        )

        if not embeddings_result:
            return []

        # DeepFace.represent returns a list of dicts; get the embedding vector
        frame_embedding = embeddings_result[0]['embedding']
        
        recognized_names = []
        
        # Compare with all known embeddings using cosine similarity
        for i, known_embedding in enumerate(known_embeddings):
            similarity = cosine_similarity(frame_embedding, known_embedding)
            
            # If similarity is above threshold, it's a match
            if similarity > threshold:
                name = known_names[i]
                if name not in recognized_names:
                    recognized_names.append(name)
                    st.write(f"🎯 Detected {name} (similarity: {similarity:.3f})")
        
        return recognized_names
        
    except Exception as e:
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
        return []

def process_video_for_attendance(video_path, camera_type="entry"):
    """Process video file and detect faces for attendance using pre-computed embeddings"""
    # Load pre-computed embeddings (FAST!)
    embeddings_data = load_embeddings_for_recognition()
    
    if not embeddings_data['embeddings']:
        st.error("No employee embeddings found. Please add employees first.")
        return []
    
    known_embeddings = embeddings_data['embeddings']
    known_names = embeddings_data['names']
    
    st.info(f"🚀 Loaded {len(known_embeddings)} pre-computed embeddings for fast recognition")
    
    cap = cv2.VideoCapture(video_path)
    detected_employees = set()
    attendance_logs = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process every 90th frame (faster since we're using pre-computed embeddings)
    frame_skip = 90
    current_frame = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_skip == 0:
            status_text.text(f"⚡ Fast processing frame {current_frame}/{total_frames}")
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)
            
            # Recognize faces using pre-computed embeddings (MUCH FASTER!)
            recognized_names = recognize_face_in_frame(small_frame, known_embeddings, known_names)
            
            for name in recognized_names:
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
    
    cap.release()
    progress_bar.empty()
    status_text.empty()
    
    return attendance_logs

def test_face_detection(image_path):
    """Test if face can be detected and compute embedding"""
    try:
        # Try to compute embedding (this also validates face detection)
        embedding = compute_embedding(image_path)
        return embedding is not None
    except:
        return False

def main():
    st.set_page_config(page_title="Employee Attendance System", layout="wide")
    st.title("🏢 Employee Attendance System (DeepFace)")
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
            st.info("💡 Tip: Upload 2-4 photos with different angles and lighting for better recognition")
            
            employee_name = st.text_input("Employee Name")
            uploaded_photos = st.file_uploader("Upload Employee Photos", 
                                              type=['jpg', 'jpeg', 'png'],
                                              accept_multiple_files=True)
            
            if st.button("Add Photos") and employee_name and uploaded_photos:
                success_count = 0
                error_count = 0
                
                with st.spinner("Processing photos and computing embeddings..."):
                    for uploaded_photo in uploaded_photos:
                        # Save uploaded photo
                        photo_dir = "employee_photos"
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                        photo_path = os.path.join(photo_dir, f"{employee_name}_{timestamp}_{uploaded_photo.name}")
                        
                        with open(photo_path, "wb") as f:
                            f.write(uploaded_photo.getbuffer())
                        
                        # Compute embedding for this photo
                        embedding = compute_embedding(photo_path)
                        
                        if embedding is not None:
                            save_employee_data(employee_name, photo_path, embedding)
                            success_count += 1
                            st.write(f"✅ Processed {uploaded_photo.name} - embedding computed")
                        else:
                            error_count += 1
                            os.remove(photo_path)  # Remove invalid photo
                            st.write(f"❌ Skipped {uploaded_photo.name} - no face detected")
                
                if success_count > 0:
                    st.success(f"✅ Added {success_count} photos for {employee_name}! Pre-computed {success_count} embeddings.")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} photos were skipped (no clear face detected)")
                    
                    # Show embedding info
                    embeddings_data = load_embeddings_for_recognition()
                    st.info(f"🚀 Total embeddings in system: {len(embeddings_data['embeddings'])}")
                    st.rerun()
                else:
                    st.error("❌ No valid faces detected in any photos. Please upload clear photos with visible faces.")
        
        with tab2:
            st.subheader("Manage Employees")
            
            # Show embedding statistics
            embeddings_data = load_embeddings_for_recognition()
            total_embeddings = len(embeddings_data['embeddings'])
            if total_embeddings > 0:
                st.info(f"🚀 System has {total_embeddings} pre-computed embeddings ready for fast recognition!")
            
            employees = get_all_employees()
            if employees:
                for emp_id, name, main_photo, photo_count, all_photos in employees:
                    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                    with col1:
                        st.write(f"**{name}**")
                        st.write(f"📸 Photos: {photo_count}")
                    with col2:
                        if main_photo and os.path.exists(main_photo):
                            st.image(main_photo, width=100, caption="Main photo")
                    with col3:
                        if st.button("View All", key=f"view_{emp_id}"):
                            st.session_state[f'show_photos_{emp_id}'] = not st.session_state.get(f'show_photos_{emp_id}', False)
                    with col4:
                        if st.button("Delete", key=f"del_{emp_id}"):
                            delete_employee(emp_id)
                            st.success(f"Employee {name} deleted!")
                            st.rerun()
                    
                    # Show all photos if expanded
                    if st.session_state.get(f'show_photos_{emp_id}', False):
                        st.write("All photos:")
                        if len(all_photos) > 0:
                            photo_cols = st.columns(min(len(all_photos), 4))
                            for i, photo_path in enumerate(all_photos):
                                if photo_path and os.path.exists(photo_path):
                                    with photo_cols[i % 4]:
                                        st.image(photo_path, width=80)
                        else:
                            st.write("No photos found")
                    st.markdown("---")
            else:
                st.info("No employees found. Please add employees first.")
    
    elif page == "Process Attendance":
        st.header("📹 Process Attendance Videos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚪 Entry Camera")
            entry_video = st.file_uploader("Upload Entry Video", 
                                         type=['mp4', 'avi', 'mov'], 
                                         key="entry")
            
            if entry_video and st.button("Process Entry Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(entry_video.getbuffer())
                    tmp_path = tmp.name
                
                st.info("🔄 Processing entry video... This may take a few minutes.")
                attendance_logs = process_video_for_attendance(tmp_path, "entry")
                os.unlink(tmp_path)
                
                if attendance_logs:
                    today = date.today().strftime("%Y-%m-%d")
                    for log in attendance_logs:
                        filename = create_attendance_log(today, log['name'], log['time'])
                    
                    st.success(f"✅ Entry processing complete! {len(attendance_logs)} employees detected.")
                    for log in attendance_logs:
                        st.write(f"🟢 {log['name']} entered at {log['time']}")
                else:
                    st.warning("⚠️ No employees detected in the entry video.")
        
        with col2:
            st.subheader("🚶‍♂️ Exit Camera")
            exit_video = st.file_uploader("Upload Exit Video", 
                                        type=['mp4', 'avi', 'mov'], 
                                        key="exit")
            
            if exit_video and st.button("Process Exit Video"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                    tmp.write(exit_video.getbuffer())
                    tmp_path = tmp.name
                
                st.info("🔄 Processing exit video... This may take a few minutes.")
                attendance_logs = process_video_for_attendance(tmp_path, "exit")
                os.unlink(tmp_path)
                
                if attendance_logs:
                    today = date.today().strftime("%Y-%m-%d")
                    for log in attendance_logs:
                        filename = create_attendance_log(today, log['name'], None, log['time'])
                    
                    st.success(f"✅ Exit processing complete! {len(attendance_logs)} employees detected.")
                    for log in attendance_logs:
                        st.write(f"🔴 {log['name']} exited at {log['time']}")
                else:
                    st.warning("⚠️ No employees detected in the exit video.")
        
        st.markdown("---")
        st.info("💡 **Processing Tips:**\n"
                "- Videos are processed every 60th frame for speed\n"
                "- Larger videos take longer to process\n"
                "- Ensure good lighting in videos for best results")
    
    elif page == "View Records":
        st.header("📊 Attendance Records")
        
        # List available attendance files
        attendance_files = [f for f in os.listdir('.') if f.startswith('attendance_') and f.endswith('.xlsx')]
        attendance_files.sort(reverse=True)  # Most recent first
        
        if attendance_files:
            selected_file = st.selectbox("Select Date", attendance_files)
            
            if selected_file:
                df = pd.read_excel(selected_file)
                
                # Display dataframe
                st.dataframe(df, use_container_width=True)
                
                # Download button
                with open(selected_file, 'rb') as f:
                    st.download_button(
                        label=f"📥 Download {selected_file}",
                        data=f.read(),
                        file_name=selected_file,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Summary statistics
                st.subheader("📈 Daily Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_employees = len(df)
                    st.metric("👥 Total Employees", total_employees)
                
                with col2:
                    present = len(df[df['Entry Time'].notna()])
                    st.metric("✅ Present", present)
                
                with col3:
                    if 'Total Hours' in df.columns:
                        avg_hours = df['Total Hours'].mean()
                        st.metric("⏰ Avg Hours", f"{avg_hours:.1f}" if not pd.isna(avg_hours) else "N/A")
                    else:
                        st.metric("⏰ Avg Hours", "N/A")
                
                with col4:
                    late_count = 0  # You can define what "late" means for your organization
                    st.metric("⏰ Late Arrivals", late_count)
                
                # Show employees who haven't exited yet
                incomplete = df[df['Entry Time'].notna() & df['Exit Time'].isna()]
                if not incomplete.empty:
                    st.subheader("🔄 Still in Office")
                    st.dataframe(incomplete[['Employee', 'Entry Time']], use_container_width=True)
        else:
            st.info("📋 No attendance records found. Process some videos first.")
            
            # Show sample data format
            st.subheader("📝 Sample Output Format")
            sample_df = pd.DataFrame({
                'Employee': ['John Doe', 'Jane Smith'],
                'Date': ['2025-08-27', '2025-08-27'],
                'Entry Time': ['09:15:30', '08:45:20'],
                'Exit Time': ['17:30:15', '18:00:10'],
                'Total Hours': [8.25, 9.25]
            })
            st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()

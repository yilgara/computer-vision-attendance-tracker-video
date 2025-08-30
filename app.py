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

def save_employee_data(name, photo_path):
    """Save employee data to files"""
    # Load existing data
    employees_data = load_employee_data()
    
    # Check if employee already exists
    existing_emp_id = None
    for emp_id, emp_data in employees_data.items():
        if emp_data['name'].lower() == name.lower():
            existing_emp_id = emp_id
            break
    
    if existing_emp_id:
        # Add new photo to existing employee
        employees_data[existing_emp_id]['photo_paths'].append(photo_path)
    else:
        # Create new employee
        employee_id = len(employees_data) + 1
        employees_data[employee_id] = {
            'name': name,
            'photo_paths': [photo_path]
        }
    
    # Save to file
    with open(EMPLOYEE_DATA_FILE, 'wb') as f:
        pickle.dump(employees_data, f)

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

def recognize_face_in_frame(frame, reference_photos):
    """Recognize faces in frame using DeepFace"""
    try:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame temporarily
        temp_frame_path = "temp_frame.jpg"
        cv2.imwrite(temp_frame_path, frame)
        
        recognized_names = []
        
        # Try to verify against each reference photo
        for name, photo_paths in reference_photos.items():
            for photo_path in photo_paths:
                try:
                    # Use DeepFace to verify if the same person
                    result = DeepFace.verify(
                        img1_path=temp_frame_path,
                        img2_path=photo_path,
                        model_name='VGG-Face',  # Fast and reliable model
                        distance_metric='cosine',
                        enforce_detection=False
                    )
                    
                    # If verification is successful and confidence is high
                    if result['verified'] and result['distance'] < 0.4:
                        if name not in recognized_names:
                            recognized_names.append(name)
                        break  # Found match, no need to check other photos of same person
                        
                except Exception as e:
                    continue  # Skip this comparison if it fails
        
        # Clean up temp file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
            
        return recognized_names
        
    except Exception as e:
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")
        return []

def process_video_for_attendance(video_path, camera_type="entry"):
    """Process video file and detect faces for attendance using DeepFace"""
    employees_data = load_employee_data()
    
    if not employees_data:
        st.error("No employees found. Please add employees first.")
        return []
    
    # Prepare reference photos dictionary
    reference_photos = {}
    for emp_id, emp_data in employees_data.items():
        reference_photos[emp_data['name']] = emp_data['photo_paths']
    
    cap = cv2.VideoCapture(video_path)
    detected_employees = set()
    attendance_logs = []
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process every 60th frame to speed up processing (DeepFace is slower)
    frame_skip = 60
    current_frame = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame % frame_skip == 0:
            status_text.text(f"Processing frame {current_frame}/{total_frames}")
            
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            
            # Recognize faces in frame
            recognized_names = recognize_face_in_frame(small_frame, reference_photos)
            
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
    """Test if face can be detected in image using DeepFace"""
    try:
        # Try to extract face
        face_objs = DeepFace.extract_faces(
            img_path=image_path,
            enforce_detection=False,
            detector_backend='opencv'
        )
        return len(face_objs) > 0
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
                
                with st.spinner("Processing photos..."):
                    for uploaded_photo in uploaded_photos:
                        # Save uploaded photo
                        photo_dir = "employee_photos"
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                        photo_path = os.path.join(photo_dir, f"{employee_name}_{timestamp}_{uploaded_photo.name}")
                        
                        with open(photo_path, "wb") as f:
                            f.write(uploaded_photo.getbuffer())
                        
                        # Test if face can be detected
                        if test_face_detection(photo_path):
                            save_employee_data(employee_name, photo_path)
                            success_count += 1
                        else:
                            error_count += 1
                            os.remove(photo_path)  # Remove invalid photo
                
                if success_count > 0:
                    st.success(f"✅ Added {success_count} photos for {employee_name}!")
                    if error_count > 0:
                        st.warning(f"⚠️ {error_count} photos were skipped (no clear face detected)")
                    st.rerun()
                else:
                    st.error("❌ No valid faces detected in any photos. Please upload clear photos with visible faces.")
        
        with tab2:
            st.subheader("Manage Employees")
            
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

import os
import cv2
import pandas as pd
from datetime import date
from face_recognition import recognize_face_in_frame



def create_attendance_log(date_str, employee_name, entry_time, exit_time=None, emp_id=None):
    """Create or update attendance log in Excel with employee ID"""
    filename = f"attendance_{date_str}.xlsx"
    
    # Check if file exists
    if os.path.exists(filename):
        df = pd.read_excel(filename)
    else:
        df = pd.DataFrame(columns=['Employee ID', 'Employee', 'Date', 'Entry Time', 'Exit Time', 'Total Hours'])
    
    # Check if employee already has an entry for today using emp_id
    if emp_id is not None:
        existing_entry = df[(df['Employee ID'] == emp_id) & (df['Date'] == date_str)]
    else:
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
            'Employee ID': emp_id,
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
    """Process video file and detect faces for attendance using pre-computed embeddings"""
    # Load pre-computed embeddings (FAST!)
    embeddings_data = load_embeddings_for_recognition()
    
    if not embeddings_data['embeddings']:
        st.error("No employee embeddings found. Please add employees first.")
        return []
    
    known_embeddings = embeddings_data['embeddings']
    known_names = embeddings_data['names']
    known_ids = embeddings_data.get('ids', [None]*len(known_embeddings))
    
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
            recognized = recognize_face_in_frame(small_frame, known_embeddings, known_names, known_ids)


            for r in recognized:
                emp_id = r['id']
                name = r['name']
                
                if emp_id not in detected_employees:
                    detected_employees.add(emp_id)
                    
                    # Calculate timestamp based on frame position
                    timestamp_seconds = current_frame / fps
                    hours = int(timestamp_seconds // 3600)
                    minutes = int((timestamp_seconds % 3600) // 60)
                    seconds = int(timestamp_seconds % 60)
                    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    attendance_logs.append({
                        'id': emp_id,
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

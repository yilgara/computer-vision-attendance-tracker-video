import streamlit as st
from attendance import process_video_for_attendance, create_attendance_log
from datetime import date
import tempfile, os

def show():
    st.header("📹 Process Attendance Videos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚪 Entry Camera")
      
        entry_video = st.file_uploader("Upload Entry Video", type=['mp4','avi','mov'], key="entry")
      
        if entry_video and st.button("Process Entry Video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(entry_video.getbuffer())
                tmp_path = tmp.name
              
            st.info("🔄 Processing entry video...")
            attendance_logs = process_video_for_attendance(tmp_path, "entry")
            os.unlink(tmp_path)
          
            if attendance_logs:
                today = date.today().strftime("%Y-%m-%d")
                for log in attendance_logs:
                    create_attendance_log(today, log['name'], log['time'], emp_id=log['id'])
                  
                st.success(f"✅ Entry complete! {len(attendance_logs)} employees detected")
                for log in attendance_logs:
                    st.write(f"🟢 {log['name']} entered at {log['time']}")
            else: 
              st.warning("⚠️ No employees detected")
    
    with col2:
        st.subheader("🚶‍♂️ Exit Camera")
      
        exit_video = st.file_uploader("Upload Exit Video", type=['mp4','avi','mov'], key="exit")
      
        if exit_video and st.button("Process Exit Video"):
          
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(exit_video.getbuffer())
                tmp_path = tmp.name
              
            st.info("🔄 Processing exit video...")
            attendance_logs = process_video_for_attendance(tmp_path, "exit")
            os.unlink(tmp_path)
          
            if attendance_logs:
                today = date.today().strftime("%Y-%m-%d")
              
                for log in attendance_logs:
                    create_attendance_log(today, log['name'], None, log['time'], emp_id=log['id'])
                  
                st.success(f"✅ Exit complete! {len(attendance_logs)} employees detected")
              
                for log in attendance_logs:
                    st.write(f"🔴 {log['name']} exited at {log['time']}")
                  
            else: st.warning("⚠️ No employees detected")
    
    st.markdown("---")
    st.info("💡 Processing Tips:\n- Videos processed every 60th frame\n- Ensure good lighting")

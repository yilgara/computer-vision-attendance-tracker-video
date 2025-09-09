import streamlit as st
import pandas as pd
import os 

def show():
    st.header("Attendance Records")
    
    attendance_files = [f for f in os.listdir('.') if f.startswith('attendance_') and f.endswith('.xlsx')]
    attendance_files.sort(reverse=True)
    
    if attendance_files:
        selected_file = st.selectbox("Select Date", attendance_files)
      
        if selected_file:
            df = pd.read_excel(selected_file)
            st.dataframe(df, use_container_width=True)
            
            with open(selected_file, 'rb') as f:
                st.download_button(
                    label=f"📥 Download {selected_file}",
                    data=f.read(),
                    file_name=selected_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.subheader("Daily Summary")
            col1,col2,col3,col4 = st.columns(4)
          
            with col1: 
                st.metric("👥 Total Employees", len(df))
              
            with col2: 
                st.metric("✅ Present", len(df[df['Entry Time'].notna()]))
              
            with col3: 
                avg_hours = df['Total Hours'].mean() if 'Total Hours' in df.columns else None
                st.metric("Avg Hours", f"{avg_hours:.1f}" if avg_hours else "N/A")
              
            with col4: 
                late_threshold = datetime.strptime("09:00:00", "%H:%M:%S").time()
                
                late_count = df['Entry Time'].dropna().apply(lambda x: datetime.strptime(x, "%H:%M:%S").time() > late_threshold).sum()

                st.metric("Late Arrivals", int(late_count))
            
            incomplete = df[df['Entry Time'].notna() & df['Exit Time'].isna()]
            if not incomplete.empty:
                st.subheader("🔄 Still in Office")
                st.dataframe(incomplete[['Employee','Entry Time']], use_container_width=True)
    else:
        st.info("📋 No attendance records found.")
